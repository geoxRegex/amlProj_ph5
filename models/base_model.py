import torch.nn as nn
from torchvision.models import resnet18
import clip


class FeatureExtractor( nn.Module ):
    def __init__(self):
        super( FeatureExtractor, self ).__init__()
        self.resnet18 = resnet18( pretrained=True )

    def forward(self, x):
        x = self.resnet18.conv1( x )
        x = self.resnet18.bn1( x )
        x = self.resnet18.relu( x )
        x = self.resnet18.maxpool( x )
        x = self.resnet18.layer1( x )
        x = self.resnet18.layer2( x )
        x = self.resnet18.layer3( x )
        x = self.resnet18.layer4( x )
        x = self.resnet18.avgpool( x )
        x = x.squeeze()
        if len( x.size() ) < 2:
            return x.unsqueeze( 0 )
        return x


class BaselineModel( nn.Module ):
    def __init__(self):
        super( BaselineModel, self ).__init__()
        self.feature_extractor = FeatureExtractor()

        self.category_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )
        self.classifier = nn.Linear( 512, 7 )

    def forward(self, x):
        x = self.feature_extractor( x )
        x = self.category_encoder( x )
        x = self.classifier( x )

        return x


class DomainDisentangleModel( nn.Module ):
    def __init__(self):
        super( DomainDisentangleModel, self ).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )
        self.category_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear( 512, 2 )
        self.object_classifier = nn.Linear( 512, 7 )

        self.reconstructor = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        ) 

    def forward(self, x, w1=None, w2=None, w3=None, w4=None, w5=None):
        x = self.feature_extractor( x )
        if w5 is None:
            if w1 is not None:
                x = self.category_encoder( x )
                x = self.object_classifier( x )
            elif w2 is not None:
                x = self.domain_encoder( x )
                x = self.domain_classifier( x )
            elif w3 is not None:
                x = self.category_encoder( x )
                x = self.domain_classifier( x )
            elif w4 is not None:
                x = self.domain_encoder( x )
                x = self.object_classifier( x )
            return x
        else:
            y = self.category_encoder( x )
            z = self.domain_encoder( x )
            y = y + z
            y = self.reconstructor( y )
            return y, x

class DomainGeneralizationModel( nn.Module ):
    def __init__(self):
        super( DomainGeneralizationModel, self ).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )
        self.category_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear( 512, 2 )
        self.object_classifier = nn.Linear( 512, 7 )

        self.reconstructor = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        ) 

    def forward(self, x, w1=None, w2=None, w3=None, w4=None, w5=None):
        x = self.feature_extractor( x )
        if w5 is None:
            if w1 is not None:
                x = self.category_encoder( x )
                x = self.object_classifier( x )
            elif w2 is not None:
                x = self.domain_encoder( x )
                x = self.domain_classifier( x )
            elif w3 is not None:
                x = self.category_encoder( x )
                x = self.domain_classifier( x )
            elif w4 is not None:
                x = self.domain_encoder( x )
                x = self.object_classifier( x )
            return x
        else:
            y = self.category_encoder( x )
            z = self.domain_encoder( x )
            y = y + z
            y = self.reconstructor( y )
            return y, x


class CLIPDisentangleModel( nn.Module ):
    def __init__(self):
        super( CLIPDisentangleModel, self ).__init__()
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu')
        #self.clip_model.eval()

        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )
        self.category_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear( 512, 2 )
        self.object_classifier = nn.Linear( 512, 7 )

        self.reconstructor = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        ) 

    def forward(self, x, y=None, w1=None, w2=None, w3=None, w4=None, w5=None,t=None):
        x = self.feature_extractor( x )
        if y is None:
            if w5 is None:
                if w1 is not None:
                    x = self.category_encoder( x )
                    x = self.object_classifier( x )
                elif w2 is not None:
                    x = self.domain_encoder( x )
                    x = self.domain_classifier( x )
                elif w3 is not None:
                    x = self.category_encoder( x )
                    x = self.domain_classifier( x )
                elif w4 is not None:
                    x = self.domain_encoder( x )
                    x = self.object_classifier( x )
                return x
            else:
                y = self.category_encoder( x )
                z = self.domain_encoder( x )
                y = y + z
                y = self.reconstructor( y )
                return y, x
        else:
            x = self.domain_encoder( x )
            text_features = self.clip_model.encode_text(y)
            return x, text_features

    
class DomainGeneralizationModel( nn.Module ):
    def __init__(self):
        super( DomainGeneralizationModel, self ).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )
        self.category_encoder = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear( 512, 2 )
        self.object_classifier = nn.Linear( 512, 7 )

        self.reconstructor = nn.Sequential(
            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),

            nn.Linear( 512, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU()
        ) 

    def forward(self, x, w1=None, w2=None, w3=None, w4=None, w5=None):
        x = self.feature_extractor( x )
        if w5 is None:
            if w1 is not None:
                x = self.category_encoder( x )
                x = self.object_classifier( x )
            elif w2 is not None:
                x = self.domain_encoder( x )
                x = self.domain_classifier( x )
            elif w3 is not None:
                x = self.category_encoder( x )
                x = self.domain_classifier( x )
            elif w4 is not None:
                x = self.domain_encoder( x )
                x = self.object_classifier( x )
            return x
        else:
            y = self.category_encoder( x )
            z = self.domain_encoder( x )
            y = y + z
            y = self.reconstructor( y )
            return y, x


