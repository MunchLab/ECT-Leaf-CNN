import numpy as np
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

# Test if this speeds up the ECT calculation on hpcc
from multiprocessing import cpu_count, Pool

# from itertools import starmap

from dataloaders import create_data_loaders, create_datasets
from utils import save_model, save_plots, save_cf, SaveBestModel, find_numpy_files
from models import CNN

from sklearn.metrics import roc_curve, auc

# Functions required for training.
def train(
    model, 
    train_loader, 
    optimizer, 
    lossfcn,
    device, 
    log_level='INFO'
):
    model.train()

    log_level = log_level == True or str(log_level).upper() == 'INFO'
    if log_level:
        print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    log_tqdm = lambda x: ( tqdm(x, total=len(train_loader)) if log_level else x, )[0]
    for data in log_tqdm(train_loader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = lossfcn(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

# function for validation
def validate(
    model, valid_loader, lossfcn, 
    device,
    log_level='INFO'
    ):
    model.eval()
    if log_level:
        print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        outputs_list = []
        labels_list = []
        for data in valid_loader:
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            outputs_list.append(outputs)
            labels_list.append(labels)
            # calculate the loss
            loss = lossfcn(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))

    return epoch_loss, epoch_acc


# Compute the ECT for given numpy file.
def compute_ect(class_name, file_path, num_dirs, num_thresh, out_file=None, log_level="INFO"):
    if isinstance(file_path, np.ndarray):
        target_file = None
    else:
        target_file = os.path.join(str(out_file), class_name, os.path.basename(file_path))
    out_file = target_file if out_file is not None else None
    
    if os.path.exists(str(target_file)) and out_file is not None:
        return class_name, np.load(out_file)
    elif os.path.exists(str(target_file)):
        return

    from ect import ECT, EmbeddedGraph

    if log_level:
        print(f'Computing ECT for {class_name}')

    G = EmbeddedGraph()
    coords = file_path if isinstance(file_path, np.ndarray) else np.load(file_path)
    G.add_cycle(coords)
    G.set_PCA_coordinates( center_type='min_max', scale_radius=1 )
    
    ect = ECT(num_dirs=num_dirs, num_thresh=num_thresh)
    ect.set_bounding_radius(1)
    val = ect.calculateECT(G)
    
    if out_file is None:
        return class_name, val
    else:
        np.save(out_file, val)

# Function to generate the ect dataset.
def generate_ect_dataset(num_dirs,num_thresh, in_path, out_path='example_data/ect_output/', parallel=True, n_workers=None, in_memory=False, log_level='INFO'):
    '''
    Generate the ECT dataset for the given input data.
    The input data should be in numpy format.
    Usage:
        generate_ect_dataset(num_dirs,num_thresh, in_path, out_path='example_data/ect_output/', global_bound_radius=2.9092515639765497, in_memory=False, log_level='INFO')
    Parameters:
        num_dirs: int, number of directions for ECT calculation.
        num_thresh: int, number of thresholds for ECT calculation.
        in_path: str, path to the input data directory.
        out_path: str, path to save the ECT dataset. Optional, default is 'example_data/ect_output/'.
        Parallel: bool, If True, use parallel processing to compute ECTs. Optional, default is True. requires additional n_workers argument. Defaults to using all available cpu cores.
        in_memory: bool,
            If True, the calculated ECT dataset is returned as a dictionary with class names as key and list of numpy arrays as the values.
            ECTs are not written to files. Optional, default is False.
        log_level: str or bool, if True or 'INFO', print progress messages
    '''

    log_level = log_level == True or str(log_level).upper() == 'INFO'

    if type(in_path) == dict:
        # If input is already a dictionary, we assume that the dict values is a list of numpy file path.
        input_numpy_files = in_path 
    else:
        # Use top level directory names as class names.
        classes = [ 
            os.path.basename(d) # Remove the path and get only the directory name.
                for d in os.listdir(in_path) # List all the directories in the input path.
                if os.path.isdir(os.path.join(in_path, d)) # Filter only directories.
        ]
        if log_level:
            print(f'Found {len(classes)} classes in the input directory.')
        
        input_numpy_files = {
            class_name: find_numpy_files(os.path.join(in_path, class_name))
            for class_name in classes
        }
    
    if in_memory:
        out_file_root = None
    else:
        out_file_root = out_path
        for class_name, f_path in input_numpy_files.items():
            os.makedirs(
                os.path.join(out_path, class_name),
                exist_ok=True
            )
    
    log_levels = [ [ log_level if i == 0 else False for i,_  in enumerate(files)] for files in input_numpy_files.values()]
    ect_arguments = [
        (class_name, file_path, num_dirs, num_thresh, out_file_root, ll)
            for lls, (class_name, files) in zip( log_levels, input_numpy_files.items() )
            for ll, file_path in zip(lls, files)
    ]            

    if not parallel:
        ects = [ compute_ect(*i) for i in ect_arguments ]
    else:
        n_workers = n_workers if n_workers is not None else cpu_count()
        with Pool(n_workers) as pool:
            ects = pool.starmap(compute_ect, ect_arguments)
    
    if in_memory:
        data = {a:b for a,b in ects}
        return data
    else:
        existing_class_items = {
            class_name: find_numpy_files(os.path.join(out_path, class_name))
                for class_name in input_numpy_files.keys() 
        }

        extra_class_items = { d:find_numpy_files(os.path.join(out_path, d)) for d in os.listdir(out_path) if d not in input_numpy_files }
        print( "Found extra class items:", list(extra_class_items) )
        print( "Removing them!")

        for files in extra_class_items.values():
            for file in files:
                os.remove(file)

        for class_name, files in existing_class_items.items():
            basename_files = [ os.path.basename(file) for file in input_numpy_files.get(class_name,[]) ]
            for file in files:
                if os.path.basename(file) not in basename_files:
                    print( "Removing:", file )
                    os.remove(file)
                else:
                    print( file, basename_files)

# model, valid_loader, lossfcn
def report_trained_model(
        num_dirs, num_thresh,
        train_dataset, train_loader, test_loader, test_dataset,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_path= 'outputs/best_model.pth',
        ax=None,
        output_cf='outputs/confusion_matrix.png',
        output_report='outputs/outputCLFreport.csv',
        log_level='INFO'
):
    """
    Function to report the trained model.
    Usage:
        report_trained_model(
            num_dirs, num_thresh,
            train_dataset, train_loader, test_loader, test_dataset,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            model_path= 'outputs/best_model.pth',
            output_cf='outputs/confusion_matrix.png',
            output_report='outputs/outputCLFreport.csv',
            log_level='INFO'
        )
    Parameters:
        num_dirs: int, number of directions for ECT calculation.
        num_thresh: int, number of thresholds for ECT calculation.
        train_dataset: torch.utils.data.Dataset, training dataset.
        train_loader: torch.utils.data.DataLoader, training data loader.
        test_loader: torch.utils.data.DataLoader, test data loader.
        test_dataset: torch.utils.data.Dataset, test dataset.
        device: torch.device, device to run the model. Optional, default is 'cuda' if available else 'cpu'.
        model_path: str, path to the trained model. Optional, default is 'outputs/best_model.pth'.
        ax: Axes, matplotlib figure axis to plot the confusion matrix on.
        output_cf: str, path to save the confusion matrix plot. Optional, default is 'outputs/confusion_matrix.png'.
        output_report: str, path to save the classification report. Optional, default is 'outputs/outputCLFreport.csv'.
        log_level: str or bool, if True or 'INFO', print progress messages.
    """
    log_level = log_level == True or str(log_level).upper() == 'INFO'
    trainimages, trainlabels = next(iter(train_loader))
    model = CNN(num_classes=train_dataset.num_classes, num_channels=trainimages.shape[1],input_resolution=(num_dirs,num_thresh))
    if log_level:
        print(model)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    if log_level:
        print('Using validation to compute confusion matrix')
    valid_running_pred = []
    valid_running_labels = []
    counter = 0
    with torch.no_grad():
        log_tqdm = lambda x: ( tqdm(x, total=len(test_loader)) if log_level else enumerate(x), )[0]
        for i, data in log_tqdm(test_loader):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            valid_running_pred.append(preds)
            valid_running_labels.append(labels)
        
    # confusion matrix for the complete epoch
    valid_running_pred = torch.cat(valid_running_pred)
    valid_running_labels = torch.cat(valid_running_labels)
    if log_level:
        print('classes:',test_dataset.classes)
    if ax is None:
        ax = plt.figure().add_subplot(111)
    save_cf(valid_running_pred.cpu(),valid_running_labels.cpu(), test_dataset.classes, ax=ax, log_level=log_level, out_cf=output_cf, out_report=output_report)

def ect_train_validate(
        num_dirs, num_thresh, input_path=None, 
        output_ect_path="example_data/ect_output", in_memory=False,
        output_model_path="outputs/best_model.pth",
        num_epochs=50, learning_rate=1e-3, lossfcn=nn.CrossEntropyLoss(),
        parallel=False, 
        batch_size=4, valid_split=0.2, num_workers=0,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        recompute_ect=True,
        log_level='INFO'
):
    """
    Function to train and validate the CNN model using the ECT dataset.
    Usage:
        ect_train_validate(
            num_dirs, num_thresh, input_path=None,
            output_ect_path="example_data/ect_output", in_memory=False,
            output_model_path="outputs/best_model.pth",
            num_epochs=50, learning_rate=1e-3, lossfcn=nn.CrossEntropyLoss(),
            batch_size=4, valid_split=0.2, num_workers=0,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            recompute_ect=True, log_level='INFO'
        )
    Parameters:
        num_dirs: int, number of directions for ECT calculation.
        num_thresh: int, number of thresholds for ECT calculation.
        input_path: str, path to the input data directory.
        output_ect_path: str, path to save the ECT dataset. Optional, default is 'example_data/ect_output'.
        output_model_path: str, path to save the trained model. Optional, default is 'outputs/best_model.pth'.
        in_memory: bool, if True, the calculated ECT dataset is stored in memory and never written to files. Optional, default is False.
        parallel: bool, if True, use parallel processing to compute ECTs. Optional, default is False.
        num_epochs: int, number of epochs for training. Optional, default is 50.
        learning_rate: float, learning rate for the optimizer. Optional, default is 1e-3.
        lossfcn: torch.nn loss function, loss function for the model. 
            Optional, default is nn.CrossEntropyLoss().
        batch_size: int, 
            Number of data loaded per epoch for training. Optional, default is 4.
            Higher batch size will require more memory and may speed up the training. Howerver, it may cause the model to be stuck in local minima.
            Lower batch size will require less memory and may slow down the training. However, it may help the model to escape local minima.
            Smaller the batch size, the more epochs are required to train the model.
        valid_split: float, 
            Fractional ratio of data to set aside for validation. Optional, default is 0.2.
        num_workers: int, 
            Number of workers for the data loader. Optional, default is 0.
            Represents the number of cpu cores used for data loading.
            This is useful for speeding up the data loading step specially when using gpu to train the model.
            However, it requires more memory.
        device: torch.device, 
            Device to run the model.
        recompute_ect: bool, 
            If True, compute the ECT dataset. Optional, default is True.
            If False, the out_path is used as the precomputed ECT dataset.
        log_level: str or bool,
            If True or 'INFO', print progress messages.
    """
    if recompute_ect:
        if log_level:
            print('Computing ECT dataset by clearing out the output directory')
        try:
            for directory in os.listdir(output_ect_path):
                directory = os.path.join(output_ect_path, directory)
                if os.path.isdir(directory):
                    for files in find_numpy_files(directory):
                        os.remove(files)
                try:
                    os.rmdir(directory)
                except:
                    pass
        except:
            Warning('Could not clear the output directory. This might cause issues with CNN training.')
        data = generate_ect_dataset(
            num_dirs, num_thresh, input_path, in_memory=in_memory, out_path=output_ect_path, log_level=log_level, parallel=parallel, n_workers=num_workers
        )
        data = data if in_memory else output_ect_path
    else:
        data = generate_ect_dataset(
            num_dirs, num_thresh, input_path, in_memory=in_memory, out_path=output_ect_path, log_level=log_level, parallel=parallel, n_workers=num_workers
        )
        data = data if in_memory else output_ect_path
    
    log_level = log_level == True or str(log_level).upper() == 'INFO'

    train_dataset, test_dataset = create_datasets(data, valid_split, log_level)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size, num_workers)
    trainimages, trainlabels = next(iter(train_loader))

    if len(trainimages) < 9:
        x,y = 2,2
    else:
        x,y = 3,3
    fig, axes = plt.subplots(x,y)
    axes = axes.flatten()
    [ (ax.get_xaxis().set_visible(False),ax.get_yaxis().set_visible(False)) for ax in axes ]
    [ s.set_visible(False) for ax in axes for s in ax.spines.values() ]
    images = np.random.choice( len(trainimages), x*y, replace=False )
    for i in range(x*y):
        axes[i].imshow( trainimages[images[i]].numpy().reshape(num_dirs, num_thresh ).T )
        axes[i].set_title(train_dataset.get_label(trainlabels[images[i]].item()))
    fig.tight_layout()
    fp = os.path.dirname(output_model_path)
    fig.savefig(f'{fp}/sample_data.png')
    
    model = CNN(num_classes=train_dataset.num_classes, num_channels=trainimages.shape[1],input_resolution=(num_dirs,num_thresh)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    save_best_model = SaveBestModel(log_level=log_level)
    train_loss, valid_loss = [],[]
    train_acc, valid_acc = [],[]

    # begin training
    for epoch in range(1,num_epochs+1):
        if log_level:
            print(f"[INFO]: Epoch {epoch} of {num_epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, lossfcn, device, log_level)
        valid_epoch_loss, valid_epoch_acc = validate(model, test_loader, lossfcn, device, log_level)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        if log_level:
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        # save the best model up to current epoch, if we have the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, lossfcn, output_model_path=output_model_path
        )
        if log_level:
            print('-'*50)
    output = {
        "num_epochs": num_epochs,
        "model": model,
        "optimizer": optimizer,
        "lossfcn": lossfcn,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset
    }
    return output

def plot_roc_curve(model, test_loader, test_dataset, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), axis=None, output_path="outputs/roc_curve.png"):
    """
    Function to plot the ROC curve for the trained model.
    Usage:
        plot_roc_curve(model, test_loader, test_dataset, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    Parameters:
        model: torch.nn model, trained model.
        test_loader: torch.utils.data.DataLoader, test data loader.
        test_dataset: torch.utils.data.Dataset, test dataset.
        device: torch.device, device to run the model. Optional, default is 'cuda' if available else 'cpu'.
    """
    from sklearn.metrics import roc_curve, auc
    if axis is None:
        ax = plt.figure().add_subplot(111)
    model.eval()
    model = model.to(device)
    print('Using validation to compute ROC curve')
    valid_running_pred = []
    valid_running_labels = []
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            valid_running_pred.append(outputs)
            valid_running_labels.append(labels)

    # confusion matrix for the complete epoch
    valid_running_pred = torch.cat(valid_running_pred)
    valid_running_labels = torch.cat(valid_running_labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(test_dataset.classes)):
        fpr[i], tpr[i], _ = roc_curve(valid_running_labels.cpu().numpy() == i, valid_running_pred.cpu().numpy()[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    for i in range(len(test_dataset.classes)):
        ax.plot(fpr[i], tpr[i],
                lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(test_dataset.classes[i], roc_auc[i]))
        
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower left',bbox_to_anchor=(1,0))
    ax.get_figure().savefig(output_path, dpi=300, bbox_inches='tight')
    return ax