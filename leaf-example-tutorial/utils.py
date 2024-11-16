import torch
import matplotlib.pyplot as plt
import os
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import LogNorm
import seaborn as sn
import numpy as np
import random

plt.style.use('ggplot')

# # Make outputs directory

# if not os.path.exists('outputs'):
#     os.makedirs('outputs')



def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class SaveBestModel:
    """
    A class that saves the best model during training by comparing current epoch validation loss to the existing lowest validation loss.
    Adapted from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    def __init__(
        self, best_valid_loss=float('inf'), log_level="INFO", output_model_path = 'outputs/best_model.pth'
    ): #initialize to infinity so the model loss must be less than existing lowest valid loss
        self.best_valid_loss = best_valid_loss
        self.log_level = log_level == True or str(log_level).upper() == 'INFO'
        self.output_model_path = output_model_path
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, output_model_path=None
    ):
        if output_model_path is None:
            output_model_path = self.output_model_path
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            if self.log_level:
                print(f"\nBest validation loss: {self.best_valid_loss}")
                print(f"\nSaving best model for epoch: {epoch}\n")
            #save model 
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                         'loss': criterion,}, output_model_path)
            

def save_model(epochs, model, optimizer, criterion, output_model_path = 'outputs/best_model.pth'):
    """
    Function to save the trained model. 
    Adapted from https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, output_model_path)
    
def save_plots(
      train_acc, valid_acc, train_loss,valid_loss,
      accuracy = None, loss = None,
      fig_size=(10, 7), dpi=300,
      accuracy_path = 'outputs/accuracy.png', loss_path = 'outputs/loss.png'):
    """
    Function to save the loss and accuracy plots.
    Usage:
        save_plots(
            train_acc, valid_acc, train_loss,valid_loss,
            accuracy = None, loss = None,
            fig_size=(10, 7), dpi=300,
            accuracy_path = 'outputs/accuracy.png', loss_path = 'outputs/loss.png'
        )
    Parameters:
        train_acc: list of training accuracy values
        valid_acc: list of validation accuracy values
        train_loss: list of training loss values
        valid_loss: list of validation loss values
        accuracy: matplotlib axis to plot accuracy. If None, a new figure is created.
        loss: matplotlib axis to plot loss. If None, a new figure is created.
        fig_size: tuple, size of the figure. Default is (10, 7)
        dpi: int, resolution of the figure. Default is 300
        accuracy_path: str, path to save the accuracy plot. Default is 'outputs/accuracy.png'
        loss_path: str, path to save the loss plot. Default is 'outputs/loss.png'
    """
    # accuracy plots
    if accuracy is None:
        accuracy = plt.figure(figsize=fig_size).add_subplot(111)
    accuracy.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    accuracy.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    accuracy.set_xlabel('Epochs')
    accuracy.set_ylabel('Accuracy')
    accuracy.legend()
    accuracy.set_title('Max validation accuracy: '+ "%.2f%%" % (np.max(valid_acc)))
    accuracy.get_figure().savefig(accuracy_path, dpi=dpi, bbox_inches='tight')
    
    # loss plots
    if loss is None:
        loss = plt.figure(figsize=fig_size).add_subplot(111)
    loss.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    loss.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    loss.set_xlabel('Epochs')
    loss.set_ylabel('Loss')
    loss.legend()
    loss.get_figure().savefig( loss_path, dpi=300, bbox_inches='tight')
    return accuracy, loss

def save_cf(
      y_pred,y_true,classes,
      ax=None,
      out_cf = 'cf_norm_logscale.png',
      out_report='outputCLFreport.csv',
      log_level='INFO'):
    """
    Function to save the confusion matrix plots.
    Usage:
        save_cf(
            y_pred,y_true,classes,
            ax=None,
            out_cf = 'cf_norm_logscale.png',
            out_report='outputCLFreport.csv',
            log_level='INFO'
        )
    Parameters:
        y_pred: list of predicted labels
        y_true: list of true labels
        classes: list of class names
        ax: matplotlib axis to plot the confusion matrix. If None, a new figure is created.
        out_cf: str, path to save the confusion matrix plot. Default is 'cf_norm_logscale.png'
        out_report: str, path to save the classification report. Default is 'outputCLFreport.csv'
        log_level: str, log level. Default is 'INFO'
    """
    cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')

    log_level = log_level == True or str(log_level).upper() == 'INFO'
    

    #df_norm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],columns = [i for i in classes])
    if ax is None:
        ax = plt.figure().add_subplot(111)
    sn.heatmap(cf_matrix, annot=True, fmt=".3f", cmap = 'Blues', norm=LogNorm(),xticklabels=classes,yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', weight='bold')
    ax.set_ylabel('True Label', weight='bold')
    plt.savefig(out_cf, dpi=300, bbox_inches='tight')
    # SAVE THE CLF REPORT   
    clf_report = pd.DataFrame(classification_report(y_true,y_pred, output_dict=True))
    clf_report.to_csv(out_report)
    if log_level:
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_true,y_pred,) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_true,y_pred,)}\n")


## TRANSFORMS ####
    
class ROTATE_TRANSLATE(object):
    """Transform the ECT matrix by randomly rotating the input image (randomly select a rotation angle) and translating the columns of the ECT image accordingly.
    """

    def __call__(self, sample):
        # get random rotation angle, use to determine how many columns to shift
        r_angle = random.uniform(0, 2*np.pi)

        cols = np.linspace(0,2*np.pi, 32)
        col_idx = min(range(len(cols)), key=lambda i: abs(cols[i]-r_angle))
        #Translate columns of image according to the random angle
        first = sample[:,col_idx:,:]
        second = sample[:,0:col_idx,:]
        new_image = torch.concatenate((first,second), axis=1)


        """         plt.style.use('default')
        fig, axes = plt.subplots(figsize=(10,5), ncols=2)
        axes[0].imshow(sample[0,:,:], cmap='gray')
        axes[1].imshow(new_image[0,:,:], cmap='gray')
        plt.show()  """   
        return new_image

    