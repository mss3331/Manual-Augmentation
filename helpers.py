import time
import copy
import torch
import numpy as np
import sklearn.metrics as sk
import pandas
import random
import Data_Related_Methods
import math
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SequentialSampler


class SubsetSampler(torch.utils.data.SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
#class ConcatDataset_myImplementation():
#
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def which_parameter_to_optimize(model_ft, feature_extract, print_names=False):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    if print_names: print('Params to learn:')

    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if print_names:
                    print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                if print_names:
                    print("\t", name)

    return params_to_update


def load_dataset_general(train_dataset,train_dataset_not_augmented, valid_test_dataset, sampler_dic,concatenate_dataset,batch_size):
    '''sampler_dic should contain train_sampler, valid_sampler, test_sampler_1, test_sampler_2 ...'''
    #batch_size = 50
    num_workers = 4
    pin_memory = False
    dataloaders_dict = {}
    dataSize_dict = {}
    train_dataset_concat = train_dataset

    if concatenate_dataset:
        train_dataset_concat = ConcatDataset([train_dataset, train_dataset_not_augmented])

    train_loader = torch.utils.data.DataLoader(
        train_dataset_concat, batch_size=batch_size["target_batch_size"], sampler=sampler_dic['train_sampler'],
        num_workers=num_workers, pin_memory=True, shuffle= False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_test_dataset, batch_size=batch_size["effective_batch_size"], sampler=sampler_dic['valid_sampler'],
        num_workers=num_workers, pin_memory=True,shuffle= False
    )

    print("Concat Dataset length = "+str(len(train_loader.sampler)))
    size_training = len(sampler_dic['train_sampler'])
    print('training images:', size_training)
    size_valid = len(sampler_dic['valid_sampler'])
    print('valid images:', size_valid)

    dataloaders_dict['train'] = train_loader
    dataloaders_dict['val'] = valid_loader
    dataSize_dict['train'] = size_training
    dataSize_dict['val'] = size_valid

    counter = 1
    for key in sampler_dic:
        if key == 'train_sampler' or key == 'valid_sampler':
            continue
        size_test = len(sampler_dic[key])
        print(key+' images:', size_test)
        loader =torch.utils.data.DataLoader(valid_test_dataset, batch_size=batch_size["effective_batch_size"],
                                            sampler=sampler_dic[key],num_workers=num_workers,
                                            pin_memory=pin_memory,shuffle= False)
        dataloaders_dict['test'+str(counter)]= loader
        dataSize_dict['test'+str(counter)] = len(sampler_dic[key])
        counter+=1



    #show_sample_images(train_loader)

    return list(dataloaders_dict.keys()),  dataloaders_dict, dataSize_dict

def getSampler(num_train, train_percentage, shuffle, dataset_num, val_percentage,concatenate_dataset=False, read_from_file=None, number_of_test_sets=10):
    '''
    Get Sampler will create list of indecies in order to be fed to the sampler.
    The sampler would be attached to torch.utils.data.DataLoader in order to fitch specific images
    :returns
    train sampler
    valid sampler
    test sampler
    '''

    assert train_percentage+val_percentage<1, "I can't split the database, training+val+test percentage should be = 1"
    indices = list(range(num_train))
    #num_train = num_train//2

    split_train_val = int(np.floor(train_percentage * num_train)) # number of images for the training
    split_val_testing = int(np.floor(val_percentage * num_train)) # number of images for the validation
    #shuffle the indeces of the images to create different dataset
    if shuffle:
        seed = random.randint(0,100000) #int(np.random.rand() * 1000)
        np.random.seed(seed)
        np.random.seed(seed)
        #print('The seed is =======',seed)
        np.random.shuffle(indices)
        np.random.seed(0)  # return back to the original
        print('indecies are shuffeled')
    else: print('indecies are not shuffeled')

    if read_from_file:# True means that there are predefined indices in a folder
        df = pandas.read_csv(read_from_file, dtype='int')
        indices = df['Dataset'+str(dataset_num)].values
        print('Reading database indecies from file '+read_from_file)

    # save the current dataset indices
    if not read_from_file:
        np.savetxt("./Figures/RandomSplit/dataset_" + str(dataset_num) + "_TrainPercentage_" + str(train_percentage) + "_valPercentage_" + str(val_percentage) + ".csv", indices, fmt='%d')

    # Extract the train\validation indeces and create the samplers
    train_idx = indices[:split_train_val]
    if concatenate_dataset:#if you are going to concat a dataset you need to modify the corresponding indecies
        dataset_size = len(indices)
        train_idx = addAugmentationIndecies(train_idx,dataset_size)

    valid_idx = indices[split_train_val: split_train_val+split_val_testing]
    train_sampler = SubsetSampler(train_idx)
    valid_sampler = SubsetSampler(valid_idx)

    # add the train\val samplers to dictionary
    all_samplers_dic = {'train_sampler': train_sampler, 'valid_sampler': valid_sampler}

    un_used_images = len(indices) - (split_train_val+split_val_testing)
    print('Un used images=',un_used_images)

    # Extract several tests from the rest of the un used images. This equation is based on the assumption that the
    # test sets would have the same size as the validation set
    start_index =split_train_val+split_val_testing
    counter = 1
    for i in range(un_used_images//split_val_testing):
        if counter>number_of_test_sets:
            break
        test_idx = indices[start_index:start_index+split_val_testing]
        start_index += split_val_testing
        all_samplers_dic['test_sampler_'+str(i+1)]=SubsetSampler(test_idx)
        counter +=1


    return all_samplers_dic
def addAugmentationIndecies(indecies,datasetSize):#if we want to append the correct indecies to the current one we need to know the new indice in the concatenated dataset
    augmented_indecies = np.add(indecies,datasetSize) # if the training index is 0, then the corresponding augmented index is = 4000
    new_indecies = []
    for i in range(len(indecies)):
        new_indecies.append(indecies[i])
        new_indecies.append(augmented_indecies[i])
    return new_indecies

def get_variouse_datasets_loaders(train_dataset,train_dataset_not_augmented,valid_test_dataset,samplers_dic_list,concatenate_dataset,batch_size):
    '''train_dataset is the imageFolder that contains the augmented set
    train_dataset_not_augmented is added in order to combine it with the augmented one so that we make the model trained over original image as well as augmented one
    valid_test_dataset is the same as train_dataset but without augmentation
    samplers_dic contains all the subsets samplers (train_sampler, valid_sampler, test_sampler_1, test_sampler_2 ... )'''

    variouse_datasets_loader = []
    for samplers_dic in samplers_dic_list:
        phases, dataloaders_dict, dataSize_dict = load_dataset_general(train_dataset,train_dataset_not_augmented, valid_test_dataset,samplers_dic,concatenate_dataset,batch_size )
        variouse_datasets_loader.append((dataloaders_dict, dataSize_dict ))

    return variouse_datasets_loader,phases

def train_model_manual_augmentation(model, dataloaders, criterion, optimizer,num_classes,data_sizes_dict, augmentation_type,
                                    orig_aug_ratio_dic, batch_size_dic,device,phases=['train', 'val'], num_epochs=25, is_inception=False):
    since = time.time()
    print(phases)
    test_acc_history = []
    test_loss_history = []
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    panda_training_validation_testing_results={} # this variable will contains accuracy\losse for each phase in order to replot the experiment
    predictions_labels_panda_dic = {} # this variable will save the model predictions and the corresponding labels for analysis later

    best_model_wts = copy.deepcopy(model.state_dict())

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch_num = -1
    co_occurence_val  = np.zeros((num_classes,num_classes))
    co_occurence_test = np.zeros((num_classes,num_classes))
    val_prec_rec_fs_support = (0,0,0,None)
    test_prec_rec_fs_support = (0,0,0,None)
    magnitude_factors_index = 0 #this index if putted inside the for loop, the experiment would be offline rather than online
    result_panda_dic = {}#this dictionary will save the important results and it will be saved using panda.
    skip_testSets_flag = True # this flag will make the model either skip or classify the testing set (we only need to know the accuracy of the testing sets if validation reached the highest accuracy)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase != 'train' and phase!= 'val' and phase!= 'test1' and skip_testSets_flag:# if all false, don't get the test sets results to save some time
                #these lines are added before to skip testing the test sets is to make the panda file consistent (i.e. same number of rows)
                if (phase + '_loss') in panda_training_validation_testing_results:
                    panda_training_validation_testing_results[(phase + '_loss')].append(-1)
                    panda_training_validation_testing_results[(phase + '_accuracy')].append(-1)
                else:
                    panda_training_validation_testing_results[(phase + '_loss')] = [-1]
                    panda_training_validation_testing_results[(phase + '_accuracy')] = [-1]
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            co_occurence = np.zeros((num_classes,num_classes))
            all_predections = []
            all_labels = []
            sub_batch = 0
            magnitude_factors = list(pandas.read_csv('./random_numbers.csv', index_col=0,dtype='float').to_numpy().squeeze()*2-1)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                if phase == "train":
                    inputs, labels, magnitude_factors_index = augmentBatch(inputs, labels, augmentation_type,
                                                              magnitude_factors,magnitude_factors_index,orig_aug_ratio_dic)



                all_labels = np.concatenate((all_labels, labels.data))

                #split the inputs and the labels into sub_batch and sub_labels inorder to be fed to the GPU
                sub_batchs_dic = splitIntoSubBatchs(inputs,labels,batch_size_dic)
                sub_batchs_images, sub_batchs_labels, sub_batchs_averaging_factor, total_batch_images = sub_batchs_dic.values()


                for sub_batch_index, temp_labels in enumerate(sub_batchs_labels):
                    sub_batch_inputs = sub_batchs_images[sub_batch_index]
                    sub_batch_labels = sub_batchs_labels[sub_batch_index]
                    # show the orig and aug images if 1:1 is applied or show the first and second batch images otherwise
                    # if phase == "train" and sub_batch<=1:
                    #     Data_Related_Methods.imshow(sub_batch_inputs, num_images=10)
                    #     sub_batch+=1


                    inputs = sub_batch_inputs.to(device)#this line cuase error if I don't have enough space in my GPU
                    labels = sub_batch_labels.to(device)
                    if(sub_batch_index==0):# zero the parameter gradients only before the first sub_batch
                        optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = (loss1 + 0.4*loss2) * sub_batchs_averaging_factor[sub_batch_index]
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels) * sub_batchs_averaging_factor[sub_batch_index]

                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()

                    # statistics for each sub_batch
                    running_loss += loss.item() * total_batch_images
                    running_corrects += torch.sum(preds == labels.data)
                    all_predections = np.concatenate((all_predections, preds.cpu().data))

                    if phase != 'train':#fill the co_occurence only for val and test
                        co_occurence=Data_Related_Methods.fill_co_occurence(pred=preds.cpu().numpy(),label=labels.cpu().numpy(), co_occurence=co_occurence)
                if phase == 'train':
                    optimizer.step()

            #epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # epoch_loss = running_loss / data_sizes_dict[phase]
            epoch_loss = running_loss / len(all_predections)
            # epoch_acc = running_corrects.double() / data_sizes_dict[phase]
            epoch_acc = running_corrects.double() / len(all_predections)
            #print(phase, len(dataloaders[phase].dataset))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #save all results in a panda file later
            if (phase+'_loss') in panda_training_validation_testing_results:
                panda_training_validation_testing_results[(phase+'_loss')].append(epoch_loss)
                panda_training_validation_testing_results[(phase+'_accuracy')].append(epoch_acc.item())
            else:
                panda_training_validation_testing_results[(phase + '_loss')]=[epoch_loss]
                panda_training_validation_testing_results[(phase + '_accuracy')]=[epoch_acc.item()]

            targeted_val_accuracy = 0.7  # this will control the frequency in which we test our model to test2 test3 ...etc
            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_epoch_num = epoch
                co_occurence_val = co_occurence
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                val_prec_rec_fs_support = sk.precision_recall_fscore_support(all_labels, all_predections,average='macro')
                if best_val_acc>=targeted_val_accuracy:
                    skip_testSets_flag = False # we can put here extra if condition to check if we reached 75% validation accuracy
                result_panda_dic['train_accuracy'] = train_acc_history[-1].item()
                result_panda_dic['val_accuracy'] = best_val_acc.item()
                result_panda_dic['val_prec_rec_fs_support'] = str(val_prec_rec_fs_support[:-1])
            elif phase == 'val':
                skip_testSets_flag = True

            if phase == 'test1' and best_epoch_num == epoch:
                best_test_acc = epoch_acc
                co_occurence_test = co_occurence
                test_prec_rec_fs_support = sk.precision_recall_fscore_support(all_labels, all_predections, average='macro')
                result_panda_dic['test1_accuracy'] = best_test_acc.item()
                result_panda_dic['test1_prec_rec_fs_support'] = str(test_prec_rec_fs_support[:-1])
            if phase != 'train' and phase!= 'val' and phase!= 'test1':# we will reach here only if skip_testSets_flag = True (i.e. validation has a new high accuracy)
                result_panda_dic[phase+'_accuracy'] = epoch_acc.item()
                result_panda_dic[phase+'_prec_rec_fs_support'] = str(sk.precision_recall_fscore_support(all_labels, all_predections, average='macro')[:-1])

            if phase != 'train' and best_epoch_num == epoch:#for val\test1\test2 .... save the model predictions and labels in order to analyze them later
                predictions_labels_panda_dic[phase + '_predictions'] = all_predections
                predictions_labels_panda_dic[phase + '_labels'] = all_labels


            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'test1':
                test_acc_history.append(epoch_acc)
                test_loss_history.append(epoch_loss)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))
    result_panda_dic['best_Epoch']=best_epoch_num

    # load best model weights
    #model.load_state_dict(best_model_wts)
    model=None
    result = {'model':model,
              'test_acc_history': test_acc_history,'test_loss_history': test_loss_history,
              'co_occurence_test': co_occurence_test,
              'test_prec_rec_fs_support':test_prec_rec_fs_support[:-1],
              'val_acc_history':val_acc_history,'val_loss_history': val_loss_history,
              'co_occurence_val':co_occurence_val,
              'val_prec_rec_fs_support':val_prec_rec_fs_support[:-1],
              'train_acc_history': train_acc_history,'train_loss_history':train_loss_history,
              'best_val_acc': best_val_acc,'best_test_acc':best_test_acc,'best_epoch_num': best_epoch_num,
              'result_panda_dic':result_panda_dic,
              'predictions_labels_panda_dic':predictions_labels_panda_dic,
              'best_model_wts':best_model_wts,
              'best_optimizer_wts':best_optimizer_wts}

    return result,panda_training_validation_testing_results
def augment(pilo_imgs, augmentation_type,magnitude_factors,magnitude_factors_index):
    '''Augment the current batch and update aumgentation factors index and return the augmente batch
    input:
        -pilo_images
        -augmentation_type (Rotate,contrast etc)
        -magnitude_factors array of random numbers
        -magnitude_factors_index index for the current augmentation factor
    output:
        -next_random_index updated index for the upcoming mag factor
        -list of augmented tensor images'''
    image_size = pilo_imgs[0].size

    # Rotate Images
    if augmentation_type.find("Rotation") >= 0:
        pilo_imgs = [TF.rotate(image, magnitude_factors[key + magnitude_factors_index] * 180, expand=True) for
                     key, image in enumerate(pilo_imgs)]
    # Contrast
    elif augmentation_type.find("Contrast") >= 0:
        pilo_imgs = [TF.adjust_contrast(image, magnitude_factors[key + magnitude_factors_index]/2 + 1) for key, image in
                     enumerate(pilo_imgs)]
    # Translate. More work is needed for this section
    elif augmentation_type.find("Translate") >= 0:
        pilo_imgs_temp=[]
        translate_ratio = [0.3, 0.3]
        index = 0
        # for each image in the list do
        for key, image in enumerate(pilo_imgs):
            image_augmented = TF.affine(image, translate=[image_size[0] * magnitude_factors[index + magnitude_factors_index]*translate_ratio[0],
                                                 image_size[1] * magnitude_factors[index+1 + magnitude_factors_index]*translate_ratio[1]],
                               angle=0, scale=1, shear=0)
            pilo_imgs_temp.append(image_augmented)
            magnitude_factors_index+=2

        pilo_imgs = pilo_imgs_temp
        next_random_index = index

    # other augmentation needs to update the next random number wherease in Translate is already updated
    if augmentation_type.find("Translate")<0:
        # What is the index of the augmentation factor for the next upcoming batch
        next_random_index = magnitude_factors_index + len(pilo_imgs)


    #resize all augmented images
    pilo_imgs = [TF.resize(image, image_size) for image in pilo_imgs]

    return pilo_imgs, next_random_index

def augmentBatch(tensor_images, labels, augmentation_type, magnitude_factors, magnitude_factors_index,orig_aug_ratio_dic):

    pilo_imgs_orig = [TF.to_pil_image(image) for image in tensor_images] #convert tensor img to pillo
    pilo_imgs_aug = []
    orig_labels = copy.deepcopy(labels)
    for i in range(orig_aug_ratio_dic["augmentation"]):
        pilo_images_temp, magnitude_factors_index = augment(pilo_imgs_orig, augmentation_type,magnitude_factors,magnitude_factors_index)
        pilo_imgs_aug+=pilo_images_temp
        if(i>0):
            labels=torch.cat((orig_labels,labels))

    if (orig_aug_ratio_dic["original"] == 1):
        pilo_imgs_aug = pilo_imgs_orig + pilo_imgs_aug
        labels=torch.cat((orig_labels,labels))

    tensor_images = [TF.to_tensor(image) for image in pilo_imgs_aug]
    tensor_images = [TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for image in
                     tensor_images]
    tensor_images = torch.stack(tensor_images)

    return tensor_images,labels, magnitude_factors_index

def splitIntoSubBatchs(inputs,labels,batch_size_dic):
    '''This function supposed to solve the problem of low GPU capacity by splitting the current batch into sub_batchs
    Input:
        list of input tensors and list of labels and the target\effetive batch size
    Output:
        splitted input\labels into multidimentional array. Each row represent sub_batch with it's label'''
    averaging_factor_list=[]
    total_batch_images = labels.size()[0]

    sub_labels_list = torch.split(labels,batch_size_dic["effective_batch_size"])
    sub_inputs_list = torch.split(inputs,batch_size_dic["effective_batch_size"])

    for i,j in enumerate(sub_labels_list):
        sub_batch_size = sub_labels_list[i].size()[0]
        averaging_factor_list.append(sub_batch_size/total_batch_images)

    sub_batchs_dic = {"sub_inputs_list":sub_inputs_list,"sub_labels_list": sub_labels_list,
                      "averaging_factor_list":averaging_factor_list,"total_batch_images":total_batch_images}
    return sub_batchs_dic
def getModel(model_name, num_classes, feature_extract, create_new= False, use_pretrained=False):
    # to make sure that we have the same parameters across multiple Torch versions. Increase repreducability
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    if create_new:
        print("creating new model '{}' with output class {} . . .".format(model_name, num_classes))
        torch.save({"model": model.state_dict(),"input_size": input_size},"Models/"+model_name+"_"+str(num_classes)+".tar")
        return model, input_size
    else:
        print("Loading model '{}' with output class {} . . .".format(model_name, num_classes))
        checkpoint = torch.load("./Models/"+model_name+"_"+str(num_classes)+".tar")
        model.load_state_dict(checkpoint["model"])
        return model, input_size

