import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

def plot(loss_history, accuracy_history, num_epochs, phase, color, file_title,title, show=False, save=False, clear_fig=False):
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch

    plt.title(title)
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs +1), loss_history, color, label=phase + " Loss")
    plt.plot(range(1, num_epochs +1), accuracy_history, "--"+color, label=phase + " Accuracy")
    plt.ylim((0, 1.1))
    plt.xticks(np.arange(1, num_epochs +1, 40.0))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.grid(True)
    #plt.legend()
    if save:
        plt.savefig("./Figures/RandomSplit/"+file_title+".png")
    if show: plt.show()
    if clear_fig: plt.clf()

def save2text(dic,title,Vertical=True):


    # save the accuracy\loss of the training\testing to .csv file
    # np.random.seed(0)
    # arr1 = np.asarray(arr1).reshape(-1,1)
    # arr2 = np.asarray(arr2).reshape(-1,1)
    # arr3 = np.asarray(arr3).reshape(-1,1)
    # arr4 = np.asarray(arr4).reshape(-1,1)
    header = ''
    result = []
    for i,key in enumerate(dic):
        data = dic[key]
        data = np.asarray(data)
        result.append(data)
        header +=key+','

    #all = np.concatenate((arr1,arr2,arr3,arr4),1)
    if Vertical:
        result = np.transpose(result)
    np.savetxt("./Figures/RandomSplit/"+title+".csv", result, header=header,delimiter=',',fmt='%f')

def save2text_co_occurence(dic,title,fmt='%d'):
    '''This function is writting specifically for saving multiple co-occurence matrices for validation and test sets
    '''
    header = ''
    result = []
    for i,key in enumerate(dic):
        data = dic[key]
        result.append(data)
        header +=key+','

    np.savetxt("./Figures/RandomSplit/"+title+".csv", np.concatenate(result,1), header=header,delimiter=',',fmt=fmt)

def save_model(title, net):
    PATH = "./Figures/RandomSplit/"+title + '.pth'
    torch.save(net.state_dict(), PATH)

    '''
                          transforms.RandomResizedCrop(input_size),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                          '''

def show_sample_images(dataloader, num_images=5):
    if dataloader!=None:
        data_iter = iter(dataloader)
        images, labels = data_iter.next()
        for i in range(num_images):
            imshow(images[i])
def imshow(imgs,normalize=True,num_images=1):
    #plot input tensor image
    #print(img.size())
    # visualize some images
    from_tensor_to_pillo = transforms.ToPILImage()
    for i in range(num_images):
        img = imgs[i]
        if normalize:
            img = from_tensor_to_pillo(img*0.225 + 0.47)
        else:
            img = from_tensor_to_pillo(img)
        img.show()

def fill_co_occurence(pred,label,co_occurence):
    for i,data in enumerate(pred):
        co_occurence[pred[i]][label[i]] += 1
    return co_occurence
