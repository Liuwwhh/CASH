import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *inputs):
        raise NotImplementedError("forward method not implemented")
    
    def save(self, path):
        """
        Save the model state to the specified path.
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path, map_location=None, strict=True):
        """
        Load the model state from the specified path.
        """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class ImageFineModule(BasicModule):
    """
    :param y_dim: dimension of texts
    :param bit: bit number of the final binary code
    """
    def __init__(self, args):
        super(ImageFineModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x)
        return x


class ImageCoarseModule(BasicModule):
    def __init__(self, args):
        super(ImageCoarseModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x)
        return x


class TextFineModule(BasicModule):
    def __init__(self, args):
        """
        :param y_dim: dimension of texts
        :param bit: bit number of the final binary code
        """
        super(TextFineModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x)
        return x
    

class TextCoarseModule(BasicModule):
    def __init__(self, args):
        super(TextCoarseModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x)
        return x
    

class MaskModule(BasicModule):
    def __init__(self, args):
        """
        :param y_dim: dimension of texts
        :param bit: bit number of the final binary code
        """
        super(MaskModule, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.normalize(x)
        x = torch.tanh(x)
        return x


class TaskIncrementalClassifier(nn.Module):
    """
    A task incremental classifier that maintains a list of classifiers for different tasks.
    Each classifier is a linear layer that outputs a single logit for binary classification.
    """
    def __init__(self, input_dim=512):
        """
        :param input_dim: dimension of the input features
        """
        super(TaskIncrementalClassifier, self).__init__()
        self.input_dim = input_dim
        self.task_classifiers = nn.ModuleList()
        self.current_task_index = -1

    def add_new_task(self):
        """
        Add a new task classifier and freeze all previous classifiers.
        """
        new_classifier = nn.Linear(self.input_dim, 1)
        self.task_classifiers.append(new_classifier)
        self.current_task_index += 1
        self.freeze_all_but_current()

    def freeze_all_but_current(self):
        """
        Freeze all classifiers except the current one.
        """
        for i, clf in enumerate(self.task_classifiers):
            for param in clf.parameters():
                param.requires_grad = (i == self.current_task_index)

    def forward(self, x):
        """
        Forward pass through the current task classifier.
        :param x: input features of shape (batch_size, input_dim)
        :return: output probabilities of shape (batch_size, 1)
        """
        if self.current_task_index < 0:
            raise ValueError("Please add task classifiers first")
        
        current_classifier = self.task_classifiers[self.current_task_index]
        logits = current_classifier(x)
        return torch.sigmoid(logits)

    def test(self, x, task_index):
        """
        Forward pass through the specified task classifier.
        :param x: input features of shape (batch_size, input_dim)
        :param task_index: index of the task classifier to use
        :return: output probabilities of shape (batch_size, 1)
        """
        if self.current_task_index < 0:
            raise ValueError("Please add task classifiers first")
        
        current_classifier = self.task_classifiers[task_index]
        logits = current_classifier(x)
        return torch.sigmoid(logits)

    def cross_task_forward(self, x):
        """
        Forward pass through all task classifiers.
        :param x: input features of shape (batch_size, input_dim)
        :return: list of output probabilities for each task classifier
        """
        outputs = []
        for clf in self.task_classifiers:
            logits = torch.sigmoid(clf(x))
            outputs.append(logits)

        return outputs

    def get_active_classifier(self):
        """
        Get the current active task classifier.
        :return: the current task classifier
        """
        return self.task_classifiers[self.current_task_index]


class HashingModel(BasicModule):
    """
    The main model that combines image and text modules with prompt learning and task incremental classification.
    """
    def __init__(self, args):
        """
        :param args: arguments containing model configurations
        """
        super(HashingModel, self).__init__()
        self.args = args

        # Define modules
        self.image_net_fine = ImageFineModule(args)
        self.image_net_coarse = ImageCoarseModule(args)
        self.text_net_fine = TextFineModule(args)
        self.text_net_coarse = TextCoarseModule(args)
        self.mask_net = MaskModule(args)

        # Initialize prompt parameters
        if self.args.prompt_mode not in ['separate', 'share']:
            raise ValueError("prompt_mode must be 'separate' or 'share'")
        elif self.args.prompt_mode == 'separate':
            self.image_prompt_list = nn.ParameterList()
            self.text_prompt_list = nn.ParameterList()
        elif self.args.prompt_mode == 'share':
            self.prompt_list = nn.ParameterList()

        # Define attention mechanism
        self.attention = MultiheadAttention(embed_dim=args.feature_dim, num_heads=8, batch_first=True)

        # Define task incremental classifier
        self.task_classifier = TaskIncrementalClassifier(input_dim=args.feature_dim)
        self.current_task_index = -1
    
    def train_prompt_enhance(self, image, text):
        """
        :param image: [batch_size, 512]
        :param text: [batch_size, 512]"""
        task_distinctiveness_image = self.task_classifier(image)
        task_distinctiveness_text = self.task_classifier(text)

        if self.args.prompt_mode not in ['separate', 'share']:
            raise ValueError("prompt_mode must be 'separate' or 'share'")
        elif self.args.prompt_mode == 'separate':
            image_prompt = self.image_prompt_list[self.current_task_index].unsqueeze(0).repeat(image.size(0), 1, 1)
            text_prompt = self.text_prompt_list[self.current_task_index].unsqueeze(0).repeat(text.size(0), 1, 1)
        elif self.args.prompt_mode == 'share':
            image_prompt = self.prompt_list[self.current_task_index].unsqueeze(0).repeat(image.size(0), 1, 1)
            text_prompt = self.prompt_list[self.current_task_index].unsqueeze(0).repeat(text.size(0), 1, 1)
        image_prompt = self.attention(image.unsqueeze(1), image_prompt, image_prompt)[0].squeeze(1)
        text_prompt = self.attention(text.unsqueeze(1), text_prompt, text_prompt)[0].squeeze(1)

        image_feature = image + (task_distinctiveness_image * image_prompt)
        text_feature = text + (task_distinctiveness_text * text_prompt)
        image_code_fine = self.image_net_fine(image_feature)
        image_code_coarse = self.image_net_coarse(image_feature)

        mask_code = self.mask_net((image_prompt + text_prompt)/2)

        text_code_fine = self.text_net_fine(text_feature)
        text_code_coarse = self.text_net_coarse(text_feature)
        
        return image_code_fine, image_code_coarse, task_distinctiveness_image, mask_code, text_code_fine, text_code_coarse, task_distinctiveness_text, image_feature, text_feature

    def forward(self, image, text):
        """
        :param image: [batch_size, 512]
        :param text: [batch_size, 512]
        """
        image_task_distinctiveness = self.task_classifier(image)
        text_task_distinctiveness = self.task_classifier(text)

        if self.args.prompt_mode not in ['separate', 'share']:
            raise ValueError("prompt_mode must be 'separate' or 'share'")
        elif self.args.prompt_mode == 'separate':
            image_prompt = self.image_prompt_list[self.current_task_index].unsqueeze(0).repeat(image.size(0), 1, 1)
            text_prompt = self.text_prompt_list[self.current_task_index].unsqueeze(0).repeat(text.size(0), 1, 1)
            image_prompt = self.attention(image.unsqueeze(1), image_prompt, image_prompt)[0].squeeze(1)
            text_prompt = self.attention(text.unsqueeze(1), text_prompt, text_prompt)[0].squeeze(1)

        elif self.args.prompt_mode == 'share':
            prompt = self.prompt_list[self.current_task_index].unsqueeze(0).repeat(image.size(0), 1, 1)
            image_prompt = self.attention(image.unsqueeze(1), prompt, prompt)[0].squeeze(1)
            text_prompt = self.attention(text.unsqueeze(1), prompt, prompt)[0].squeeze(1)

        image_feature = image + (image_task_distinctiveness * image_prompt)
        text_feature = text + (text_task_distinctiveness * text_prompt)
        image_code_fine = self.image_net_fine(image_feature)
        image_code_coarse = self.image_net_coarse(image_feature)
        text_code_fine = self.text_net_fine(text_feature)
        text_code_coarse = self.text_net_coarse(text_feature)

        mask_code = self.mask_net((image_prompt + text_prompt)/2)

        return image_code_fine, text_code_fine, image_code_coarse, text_code_coarse, mask_code, image_task_distinctiveness, text_task_distinctiveness
    
    @torch.no_grad()
    def cross_task_forward(self, image, text):
        """
        :param image: [batch_size, 512]
        :param text: [batch_size, 512]
        :return: image_code, text_code
        """
        if self.args.prompt_mode not in ['separate', 'share']:
            raise ValueError("prompt_mode must be 'separate' or 'share'")
        elif self.args.prompt_mode == 'separate':
            image_task_distinctiveness = self.task_classifier.cross_task_forward(image)
            text_task_distinctiveness = self.task_classifier.cross_task_forward(text)

            stacked_image_task_distinctiveness = torch.stack(image_task_distinctiveness, dim=0)
            stacked_text_task_distinctiveness = torch.stack(text_task_distinctiveness, dim=0)

            stacked_image_task_distinctiveness = stacked_image_task_distinctiveness.squeeze(-1).argmax(dim=0)
            stacked_text_task_distinctiveness = stacked_text_task_distinctiveness.squeeze(-1).argmax(dim=0)

            image_prompt = []
            text_prompt = []
            for i, j in zip(stacked_image_task_distinctiveness, stacked_text_task_distinctiveness):
                image_prompt.append(self.image_prompt_list[i].unsqueeze(0))
                text_prompt.append(self.text_prompt_list[j].unsqueeze(0))
            image_prompt = torch.cat(image_prompt, dim=0)
            text_prompt = torch.cat(text_prompt, dim=0)
            enhanced_image_prompt = self.attention(image.unsqueeze(1), image_prompt, image_prompt)[0].squeeze(1)
            enhanced_text_prompt = self.attention(text.unsqueeze(1), text_prompt, text_prompt)[0].squeeze(1)

        elif self.args.prompt_mode == 'share':
            image_task_distinctiveness = self.task_classifier.cross_task_forward(image)
            text_task_distinctiveness = self.task_classifier.cross_task_forward(text)

            stacked_image_task_distinctiveness = torch.stack(image_task_distinctiveness, dim=0)
            stacked_text_task_distinctiveness = torch.stack(text_task_distinctiveness, dim=0)

            stacked_image_task_distinctiveness_index = stacked_image_task_distinctiveness.squeeze(-1).argmax(dim=0)
            stacked_text_task_distinctiveness_index = stacked_text_task_distinctiveness.squeeze(-1).argmax(dim=0)

            image_prompt = []
            text_prompt = []
            for i, j in zip(stacked_image_task_distinctiveness_index, stacked_text_task_distinctiveness_index):
                image_prompt.append(self.prompt_list[i].unsqueeze(0))
                text_prompt.append(self.prompt_list[j].unsqueeze(0))
            image_prompt = torch.cat(image_prompt, dim=0)
            text_prompt = torch.cat(text_prompt, dim=0)

            enhanced_image_prompt = self.attention(image.unsqueeze(1), image_prompt, image_prompt)[0].squeeze(1)
            enhanced_text_prompt = self.attention(text.unsqueeze(1), text_prompt, text_prompt)[0].squeeze(1)

        image_feature = image + enhanced_image_prompt
        text_feature = text + enhanced_text_prompt
        image_code_fine = self.image_net_fine(image_feature)
        image_code_coarse = self.image_net_coarse(image_feature)
        text_code_fine = self.text_net_fine(text_feature)
        text_code_coarse = self.text_net_coarse(text_feature)

        image_mask_code = self.mask_net(enhanced_image_prompt)
        text_mask_code = self.mask_net(enhanced_text_prompt)

        image_hash = torch.tanh(image_code_fine) * ((1-image_mask_code)/2) + torch.tanh(image_code_coarse) * ((1+image_mask_code)/2)
        text_hash = torch.tanh(text_code_fine) * ((1-text_mask_code)/2) + torch.tanh(text_code_coarse) * ((1+text_mask_code)/2)

        return image_hash, text_hash


    @torch.no_grad()
    def test(self, image, text, task_index):
        """
        :param image: [batch_size, 512]
        :param text: [batch_size, 512]
        :return: image_code, text_code
        """
        image_task_distinctiveness = self.task_classifier.test(image, task_index)
        text_task_distinctiveness = self.task_classifier.test(text, task_index)

        if self.args.prompt_mode not in ['separate', 'share']:
            raise ValueError("prompt_mode must be 'separate' or 'share'")
        elif self.args.prompt_mode == 'separate':
            image_prompt = self.image_prompt_list[task_index]
            text_prompt = self.text_prompt_list[task_index]
            image_prompt = image_prompt.unsqueeze(0).repeat(image.size(0), 1, 1)
            text_prompt = text_prompt.unsqueeze(0).repeat(text.size(0), 1, 1)
            image_prompt = self.attention(image.unsqueeze(1), image_prompt, image_prompt)[0].squeeze(1)
            text_prompt = self.attention(text.unsqueeze(1), text_prompt, text_prompt)[0].squeeze(1)

        elif self.args.prompt_mode == 'share':
            prompt = self.prompt_list[task_index].unsqueeze(0).repeat(image.size(0), 1, 1)
            image_prompt = self.attention(image.unsqueeze(1), prompt, prompt)[0].squeeze(1)
            text_prompt = self.attention(text.unsqueeze(1), prompt, prompt)[0].squeeze(1)

        image_feature = image + (image_task_distinctiveness * image_prompt)
        text_feature = text + (text_task_distinctiveness * text_prompt)
        image_code_fine = self.image_net_fine(image_feature)
        image_code_coarse = self.image_net_coarse(image_feature)

        text_code_fine = self.text_net_fine(text_feature)
        text_code_coarse = self.text_net_coarse(text_feature)

        image_mask_code = self.mask_net(image_prompt)
        text_mask_code = self.mask_net(text_prompt)

        image_hash = torch.tanh(image_code_fine) * ((1-image_mask_code)/2) + torch.tanh(image_code_coarse) * ((1+image_mask_code)/2)
        text_hash = torch.tanh(text_code_fine) * ((1-text_mask_code)/2) + torch.tanh(text_code_coarse) * ((1+text_mask_code)/2)
        return image_hash, text_hash

    def add_prompt(self):
        """
        Add a new prompt for a new task and a new task classifier.
        """
        self.current_task_index += 1
        if self.args.prompt_mode == 'separate':
            image_prompt = nn.Parameter(
                torch.randn(self.args.prompt_length, self.args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.image_prompt_list.append(image_prompt)
            text_prompt = nn.Parameter(
                torch.randn(self.args.prompt_length, self.args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.text_prompt_list.append(text_prompt)
        elif self.args.prompt_mode == 'share':
            prompt = nn.Parameter(
                torch.randn(self.args.prompt_length, self.args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.prompt_list.append(prompt)