from __future__ import print_function
from __future__ import division
import helpers
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import pandas
import time
import Data_Related_Methods
import torchvision
import random
#import compareTwoModel
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

def runManualAugmentation(variouse_datasets_loader, augmentaionType, model, num_classes,phases=['train', 'val']):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training will be in the "' + str(device) + '"')

    best_val_accuracies = []
    best_epochs = []
    corres_test_accuracies = []  # Accuracy taken at an epoch number correspond to highest val acuracy.
    best_val_co_occurence = []
    corres_test_co_occurence = []
    best_val_prec_rec_fs_support = []
    corres_test_prec_rec_fs_support = []
    panda_results = {}
    predictions_labels_panda_dic = {}
    exp_num = 0
    for key, (dataloaders_dict, data_sizes_dict) in enumerate(variouse_datasets_loader):
        exp_num += 1
        # re-copy the start state
        model_ft = copy.deepcopy(model)
        model_ft = model_ft.to(device)
        params_to_update = helpers.which_parameter_to_optimize(model_ft, feature_extract, False)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)#, momentum=0.9)
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        # Train and evaluate
        results = helpers.train_model_manual_augmentation(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                      num_classes,
                                      augmentation_type=augmentaionType,
                                      data_sizes_dict=data_sizes_dict,
                                      device=device,
                                      num_epochs=num_epochs,
                                      is_inception=(model_name == "inception"),
                                      phases=phases)
        (results, panda_training_validation_testing_results) = results
        # summary of the best validation\test accuarcy achieved across a specific dataset along with its co_occurence
        best_val_accuracies.append((results['best_val_acc'].item()))
        best_epochs.append(results['best_epoch_num'])
        best_val_co_occurence.append(results['co_occurence_val'])
        best_val_prec_rec_fs_support.append(results['val_prec_rec_fs_support'])
        corres_test_accuracies.append((results['best_test_acc'].item()))
        corres_test_co_occurence.append(results['co_occurence_test'])
        corres_test_prec_rec_fs_support.append(results['test_prec_rec_fs_support'])

        # save panda results
        pandas.DataFrame(panda_training_validation_testing_results).to_csv(
            './Figures/PandaResults/PandaAllResults_' + augmentaionType + '_Dataset' + str(exp_num) + '.csv')
        pandas.DataFrame(results['predictions_labels_panda_dic']).to_excel(
            './Figures/Predictions/Predictions_' + augmentaionType + '_Dataset' + str(exp_num) + '.xlsx')
        panda_results['Dataset' + str(exp_num)] = results['result_panda_dic']
        parameters_used = "Pretrained_False_Augmentation" + str(augmentaionType) + str(exp_num) \
                          + "_bestEpochParameters" + model_name + \
                          "_epochs" + str(num_epochs) + "_featureExtraction" + str(feature_extract)


        Data_Related_Methods.plot(results['val_loss_history'], results['val_acc_history'],
                     file_title=parameters_used,
                     title=augmentaionType,
                     color="r",
                     num_epochs=num_epochs,
                     phase="val",
                     show=False)
        Data_Related_Methods.plot(results['test_loss_history'], results['test_acc_history'],
                     file_title=parameters_used,
                     title=augmentaionType,
                     color="g",
                     num_epochs=num_epochs,
                     phase="test",
                     show=False)
        Data_Related_Methods.plot(results['train_loss_history'], results['train_acc_history'],
                     file_title=parameters_used,
                     title=augmentaionType,
                     color="b",
                     num_epochs=num_epochs,
                     phase="train",
                     show=False, save=True)
        print('********************** Finish ' + augmentaionType + ' :' + str(exp_num))
        # check point where created incase I want to use transfer learning
        torch.save({
            'best_epoch_num': results['best_epoch_num'],
            'best_model_wts': results['best_model_wts'],
            'best_optimizer_wts': results['best_optimizer_wts'],
            'best_val_acc': results['best_val_acc']},
            './Figures/CheckPoints/' + augmentaionType + '_' + model_name + '_Dataset' + str(exp_num) + '.pth')

    plt.clf()
    print('Best Validation Accuracies across all Datasets = ', (best_val_accuracies, best_epochs))
    print('The corresponding Test acuracies = ', corres_test_accuracies)
    pandas.DataFrame(panda_results).T.to_excel('./Figures/' + augmentaionType + '_panda..xlsx')
    # print('Best Validation Co_occurence matrix:\n',best_val_co_occurence)
    # print('The corresponding Test Co_occurence matrix:\n',corres_test_co_occurence)

    # Save data to text
    best_accuracies_dic = {'best_val_accuracies': best_val_accuracies,
                           'corres_test_accuracies': corres_test_accuracies,
                           'Epoch_number': best_epochs}
    Data_Related_Methods.save2text(best_accuracies_dic, title=parameters_used)

    best_co_occurences_dic = {'best_val_co_occurences': np.concatenate(best_val_co_occurence).squeeze(),
                              'corres_test_co_occurence': np.concatenate(corres_test_co_occurence).squeeze()}
    Data_Related_Methods.save2text_co_occurence(best_co_occurences_dic, title='co_occurences_' + parameters_used)

    best_reports_dic = {'best_val_prec_rec_fs_support': np.reshape(best_val_prec_rec_fs_support, (-1, 3)),
                        'corres_test_prec_rec_fs_support': np.reshape(corres_test_prec_rec_fs_support, (-1, 3))
                        }
    Data_Related_Methods.save2text_co_occurence(best_reports_dic, title='prec_rec_fs_supp_' + parameters_used, fmt='%f')

def get_dataset_withSpecificTransform(input_size, data_dir, specific_transform=None):
    '''This function create a dataloader with a specific transform added to the default transforms
    '''
    if specific_transform==None:
        transform = transforms.Compose(
            [transforms.Resize(input_size), transforms.CenterCrop(input_size), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose(
            [transforms.Resize(input_size), specific_transform ,transforms.Resize(input_size), transforms.CenterCrop(input_size),
             transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return datasets.ImageFolder(root=data_dir, transform=transform)

def ManualAugmentationExperiments(batch_size, model_name):
    print("Manual Augmentation is running ....")
    #data_dir = "C:/Users/Mahmood_Haithami/Downloads/JDownloader/Databases/KvasirV1_WithBlackBox_HE_0.3GB" # Windows
    data_dir = "C:/Users/Mahmood_Haithami/Downloads/JDownloader/Databases/KvasirV1_Unified" # Windows
    #data_dir = "~/Documents/Mahmood/Databases/KvasirV1_Unified"

    # Initialize the model for this run
    model, input_size = helpers.getModel(model_name, num_classes, feature_extract, create_new = False ,use_pretrained=False)

    # 1- without Augmentation
    basic_transform = transforms.Compose([transforms.Resize(input_size), transforms.CenterCrop(input_size), transforms.ToTensor()])
    train_dataset_noAugmentation = datasets.ImageFolder(data_dir, transform=basic_transform)
    valid_test_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([transforms.Resize(input_size), transforms.CenterCrop(input_size), transforms.ToTensor(),
                                                                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])) # validation and test set should be without any modification
    datasetSize = len(train_dataset_noAugmentation)  # datasetSize = 4000 for Kvasir V1
    # ------------------------------------- Finish constructing all my transforms -----------------------------

    samplers_dic_list = []  # this should contains a list of datasets (train,validation and a bunsh of test sets). The
    # size of the list represent the number of experements to be conducted
    # construct a random indices "samplers to be feed into dataloaders". This indicate the number of experiment to be run
    samplers_dic_list.append(helpers.getSampler(datasetSize, train_percentage=train_percentage, val_percentage=val_percentage, shuffle=False, dataset_num=1, read_from_file=read_from_file, concatenate_dataset=concatenate_dataset))
    # samplers_dic_list.append(helpers.getSampler(datasetSize, train_percentage=train_percentage, val_percentage=val_percentage, shuffle=True,dataset_num=2, read_from_file=read_from_file,concatenate_dataset=concatenate_dataset))

    # ----------------------------Finish Defining our variouse dataloaders ---------------------------------
    augmentations = ["No_Augmentation Manual Augmentation", "Random Rotation [-180 +180] Resized",
                     "Random Contrast [0.5 2]", "Random Translate [0.3 0.3]"]
    for augmentation_type in augmentations :
        variouse_datasets_loader=[]
        variouse_datasets_loader, phases = helpers.get_variouse_datasets_loaders(train_dataset_noAugmentation,
                                                                                 train_dataset_noAugmentation,
                                                                                 valid_test_dataset, samplers_dic_list,
                                                                                 batch_size=batch_size,
                                                                                 concatenate_dataset=concatenate_dataset)
        runManualAugmentation(variouse_datasets_loader, augmentation_type, model, num_classes, phases=phases)
        torch.manual_seed(0)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "inception"
    #which indecies to consider
    read_from_file = "./dataset1_dataset2_KvasirV1.csv"
    # train set percintage
    train_percentage = 0.25
    # validation set percintage
    val_percentage = 0.125
    # Number of classes in the dataset
    num_classes = 8
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    # Number of epochs to train for
    num_epochs = 1
    batch_size = 10
    concatenate_dataset = False
    start = time.time()
    ManualAugmentationExperiments(batch_size,model_name)
    total_time = time.time() - start
    print('-'*50,'\nThe entire experiments completed in {:.0f}h {:.0f}m'.format(total_time // 60**2, (total_time % 60**2) // 60))

