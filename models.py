# create densenet 121 model
# create densenet 161 model
# create densenet 169 model
# create densenet 201 model
# create inception_resnet_v2 model

import torch
import torchvision.models as models
import torch.nn as nn
import timm

def create_densenet121(num_classes=5, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    classifier_in_features = model.classifier.in_features
    model.classifier = nn.Linear(classifier_in_features, num_classes)

    return model

def create_densenet161(num_classes=5, pretrained=True):
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    classifier_in_features = model.classifier.in_features
    model.classifier = nn.Linear(classifier_in_features, num_classes)
    return model

def create_densenet169(num_classes=5, pretrained=True):
    model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    classifier_in_features = model.classifier.in_features
    model.classifier = nn.Linear(classifier_in_features, num_classes)
    return model

def create_densenet201(num_classes=5, pretrained=True):
    model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    classifier_in_features = model.classifier.in_features
    model.classifier = nn.Linear(classifier_in_features, num_classes)
    return model

def create_inception_resnet_v2(num_classes=5, pretrained=True):
    model = timm.create_model("inception_resnet_v2" ,num_classes = 5, pretrained=pretrained)
    return model

def create_ensemble():
    model1 = create_densenet121()
    model2 = create_densenet161()
    model3 = create_densenet169()
    model4 = create_densenet201()
    model5 = create_inception_resnet_v2()
    
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super(EnsembleModel, self).__init__()
            self.models = nn.ModuleList(models)
            self.fc = nn.Linear(len(models) * 5, 5)  # Assuming each model outputs 5 classes

        def forward(self, x):
            outputs = [model(x) for model in self.models]
            x = torch.cat(outputs, dim=1)
            x = self.fc(x)
            return x

    ensemble_model = EnsembleModel([model1, model2, model3, model4, model5])
    return ensemble_model