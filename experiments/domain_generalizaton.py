import torch
import torch.nn.functional as func
from models.base_model import DomainGeneralizationModel



def myEntropyLoss(opt):
    return torch.sum(-func.softmax(opt, 1) * func.log_softmax(opt, 1), 1).mean()


def myReconstructorLoss(reconstructorOutputs, features):
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.KLDivLoss()
    return loss1( reconstructorOutputs, features ) + loss2( reconstructorOutputs, features )


class DomainGeneralizationExperiment:  # See point 2. of the project

    def __init__(self, opt, weight=None):
        self.opt = opt
        self.device = torch.device( 'cpu' if opt['cpu'] else 'cuda:0' )
        self.weights = weight

        # Setup model
        self.model = DomainGeneralizationModel()
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=opt['lr'] )

        self.object_classifier_criterion = torch.nn.CrossEntropyLoss()
        self.domain_classifier_criterion = torch.nn.CrossEntropyLoss()
        self.domain_category_criterion = myEntropyLoss
        self.object_domain_criterion = myEntropyLoss
        self.reconstructor_criterion = myReconstructorLoss

        self.model.to( self.device )

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save( checkpoint, path )

    def load_checkpoint(self, path):
        checkpoint = torch.load( path )

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict( checkpoint['model'] )
        self.optimizer.load_state_dict( checkpoint['optimizer'] )

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, train=True, weight=None):
        self.weights = weight
        self.optimizer.zero_grad()

        if train:
            x, y, z = data
            x = x.to( self.device )
            y = y.to( self.device )
            z = z.to( self.device )

            for param in self.model.domain_encoder.parameters():
                param.requires_grad = False
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w1=1 )
            loss = self.object_classifier_criterion( logits, y ) * self.weights[0]
            loss.backward()
            for param in self.model.domain_encoder.parameters():
                param.requires_grad = True
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True

            for param in self.model.category_encoder.parameters():
                param.requires_grad = False
            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w2=1 )
            loss = self.domain_classifier_criterion( logits, z ) * self.weights[1]
            loss.backward()
            for param in self.model.category_encoder.parameters():
                param.requires_grad = True
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True

            for param in self.model.domain_encoder.parameters():
                param.requires_grad = False
            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w3=self.weights[0] )
            loss = self.domain_category_criterion( logits ) * self.weights[2]
            loss.backward()
            for param in self.model.domain_encoder.parameters():
                param.requires_grad = True
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True

            for param in self.model.category_encoder.parameters():
                param.requires_grad = False
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w4=self.weights[1] )
            loss = self.object_domain_criterion( logits ) * self.weights[3]
            loss.backward()
            for param in self.model.category_encoder.parameters():
                param.requires_grad = True
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True

            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = False
            logits, X = self.model( x, w5=self.weights[2] )
            loss = self.reconstructor_criterion( logits, X ) * self.weights[4]
            loss.backward()
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = True

        else:
            x, y, z = data
            x = x.to( self.device )
            z = z.to( self.device )

            for param in self.model.category_encoder.parameters():
                param.requires_grad = False
            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w2=1 )
            loss = self.domain_classifier_criterion( logits, z ) * self.weights[1]
            loss.backward()
            for param in self.model.category_encoder.parameters():
                param.requires_grad = True
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True
            
            for param in self.model.domain_encoder.parameters():
                param.requires_grad = False
            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.reconstructor.parameters():
                param.requires_grad = False
            logits = self.model( x, w3=self.weights[0] )
            loss = self.domain_category_criterion( logits ) * self.weights[2]
            loss.backward()
            for param in self.model.domain_encoder.parameters():
                param.requires_grad = True
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.reconstructor.parameters():
                param.requires_grad = True

            for param in self.model.object_classifier.parameters():
                param.requires_grad = False
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = False
            logits, X = self.model( x, w5=self.weights[2] )
            loss = self.reconstructor_criterion( logits, X ) * self.weights[4]
            loss.backward()
            for param in self.model.object_classifier.parameters():
                param.requires_grad = True
            for param in self.model.domain_classifier.parameters():
                param.requires_grad = True

        self.optimizer.step()

        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y, z in loader:
                x = x.to( self.device )
                y = y.to( self.device )

                logits = self.model( x, w1=1 )
                loss += self.object_classifier_criterion( logits, y )
                pred = torch.argmax( logits, dim=-1 )
                accuracy += (pred == y).sum().item()
                count += x.size( 0 )

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss
