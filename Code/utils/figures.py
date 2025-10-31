import copy
from collections import Counter
from itertools import cycle

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from albumentations.pytorch import ToTensorV2
from IPython.display import HTML, display
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from utils.data_loader import BoundingBoxTransformer, Mosaico_dataset, n_mosquito, num2string
from utils.losses_metrics import macro_acc, micro_acc
from utils.models import MOSAICO_training


def display_image_grid(X_train,train_labels, predicted_labels=(), cols=4,SAVE = ""):
    rows = 3#len(X_train) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    random_selected = np.random.choice(len(X_train), rows*cols)
    for i, random_idx in enumerate(random_selected):
        image = X_train[random_idx,:,:,:]

        true_label = train_labels[random_idx]["species"]
        source= train_labels[random_idx]["source"]
        predicted_label = predicted_labels[random_idx] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(f"{predicted_label} ({source})", color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    if SAVE:
        plt.savefig("results/"+SAVE+".jpg")
    plt.show()




def visualize_augmentations(dataset, idx=0, samples=10, cols=5,SAVE = ""):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    if SAVE:
        plt.savefig("results/"+SAVE+".jpg")
    plt.show()


def plot_history(history, results_folder_path, SAVE=""):

    epochs_train = np.arange(history.train_loss["epoch"])+1
    epochs_val = np.arange(history.validation_loss["epoch"])+1
    train_loss = history.train_loss["value"]
    val_loss = history.validation_loss["value"]
    train_acc = history.train_accuracy["value"]
    lr_curve = history.lr["value"]



    f,ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    ax.plot(epochs_train,train_loss,label="train",color="blue")
    ax.plot(epochs_val,val_loss,label="validation",color="orange")
    ax2.axvline(history.best_epoch,color="r",linestyle="--",label="best epoch")
    ax.legend(loc=1)
    ax.set_ylabel("Loss")


    ax2.plot(epochs_train, lr_curve,"k--",label="LR",)
    ax2.legend(loc=2)
    plt.xlabel("Epochs")
    ax2.set_ylabel("LR")
    ax.set_yscale("log")


    epochs_train = np.arange(history.train_accuracy["epoch"])+1
    epochs_val = np.arange(history.validation_accuracy["epoch"])+1

    val_acc = history.validation_accuracy["value"]
    train_acc = history.train_accuracy["value"]
    lr_curve = history.lr["value"]
    if SAVE:
        plt.savefig(f"{results_folder_path}/Loss_history_fold_"+SAVE)



    f,ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    ax.plot(epochs_train,train_acc,label="train",color="blue")
    ax.plot(epochs_val,val_acc,label="validation",color="orange")
    ax2.axvline(history.best_epoch,color="r",linestyle="--",label="best epoch")
    ax.legend(loc=1)
    ax.set_ylabel("Accuracy")


    ax2.plot(epochs_train, lr_curve,"k--",label="LR",)
    ax2.legend(loc=2)
    ax.set_xlabel("Epochs")
    ax2.set_ylabel("LR")
    #ax.set_yscale("log")
    if SAVE:
        plt.savefig(f"{results_folder_path}/ACC_history_fold_"+SAVE)

    # f,ax = plt.subplots(1,1)
    # reg_curve = history.reg["value"]
    # ax.plot(epochs_train, reg_curve,"k",label="L1",)
    # ax.legend(loc=2)
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("L1 + L2")



def ROC_unknown(y_true_test, y_pred_test, labels, results_folder_path, SAVE=""):
    n_classes = 2
    target_names = labels
    y_onehot_test = np.zeros((y_true_test.shape[0],n_classes))
    y_onehot_test[np.arange(y_true_test.shape[0]),y_true_test] = 1
    #y_onehot_test = ohe.transform(y_true_test.reshape(-1,1))
    y_score = y_pred_test


    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = {}, {}, {}
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["skyblue", "red"])
    for class_id, color in zip(range(n_classes), colors, strict=True):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            #plot_chance_level=(class_id == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    if SAVE:
        plt.savefig(f"{results_folder_path}/ROC_unknown_fold_"+SAVE)
    plt.show()


def ROC(y_true_test, y_pred_test, labels,results_folder_path, SAVE=""):
    n_classes = y_pred_test.shape[1]
    target_names = labels
    y_onehot_test = np.zeros((y_true_test.shape[0],n_classes))
    y_onehot_test[ np.arange(y_true_test.shape[0]),y_true_test] = 1
    #y_onehot_test = ohe.transform(y_true_test.reshape(-1,1))
    y_score = y_pred_test


    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = {}, {}, {}
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue","darkgreen","purple"])
    for class_id, color in zip(range(n_classes), colors, strict=True):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            #plot_chance_level=(class_id == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    if SAVE:
        plt.savefig(f"{results_folder_path}/ROC_fold_"+SAVE)
    plt.show()


def species_sources_Table(path, data_path, sources = False):
    Table = pd.read_excel(path)
    if sources:
        pivot_Table = Table.pivot_table(index='species', columns='source', aggfunc='size', fill_value=0)
    else:
        train_files = Table["file"].apply(lambda x: "{}{}".format(x,".npz")).values
        n_mosq_arr = n_mosquito(data_path, train_files)
        Table["n_mosquitos"] = n_mosq_arr
        pivot_Table = pd.pivot_table(Table, values='n_mosquitos', index='species', columns='source', aggfunc='sum', fill_value=0).astype(int)
    tot = pivot_Table.sum(axis=1)
    pivot_Table["tot"] = tot
    #boolean_Table = pivot_Table.notna()
    boolean_Table = pivot_Table
    boolean_Table.index.name = None
    boolean_Table.columns.name = None
    styled_df = boolean_Table.style \
        .set_properties(**{'text-align': 'center'}) \
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])

    display(HTML(styled_df.render()))

    styled_df.to_html('styled_table.html', index=False)


def get_abbr(label):
    genus = label.split(" ")[0]
    abbr = ''.join(car for (i, car) in enumerate(genus) if i<2)
    species = label.split(" ")[1]
    pred = ''.join((abbr, ". ", species))
    return pred


def segment_photo(image_path):
        # INITIALIZE CLASS TO PASS FROM ORIGINAL PHOTO TO CUTTED MOSQUITO
        label_dict = {}
        # Segment the image
        transformer = BoundingBoxTransformer(debug=True)
        new_bounding_boxes, mosquito_array, label_array, new_img, message = (
            transformer.transform_bounding_boxes(image_path, label_dict)
        )
        return new_bounding_boxes, mosquito_array, label_array, new_img, message


def image_predictions( mosquito_array, results_folder):
    
    val_transform = A.Compose(
        [

            A.CenterCrop(height=360, width=360),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    
    with np.load(f"Tot_Field_dataset/results/{results_folder}/train_test_split_files.npz", allow_pickle=True) as loaded_file:
            params = loaded_file["params"].item()
    fold = 1
    params["checkpoint"] = f"Tot_Field_dataset/results/{results_folder}/fold_{fold}.pt"
    strig2num = params["strig2num"]

    label_photo  = np.zeros(len(mosquito_array))
    photo_dataset = Mosaico_dataset(mosquito_array, label_photo, transform=val_transform)
    photo_loader = DataLoader(photo_dataset, batch_size=1, shuffle=False)#, num_workers=params["num_workers"])
    mosaico = MOSAICO_training(params)
    mosaico.load_model()

    photo_pred, _ = mosaico.test(photo_loader)
    uncertainty = photo_pred[:, -1]
    return photo_pred, uncertainty, strig2num


def elaborate_group_photo(image_path, results_folder, SHOW=False):
    init_img = cv2.imread(image_path)
    new_bounding_boxes, mosquito_array, _, image_test, log = segment_photo(image_path)
    photo_pred, uncertainty, strig2num = image_predictions(mosquito_array, results_folder)
    
    num_pred = np.argmax(photo_pred, axis=1)
    label_pred = num2string(strig2num, num_pred)

    new_img = image_test.copy()
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB) #RGB
    Table_list = []
    for i, box in enumerate(new_bounding_boxes):
        photo_dict = {}
        _, x, y, w, h = box
        photo_dict["bbox_x"] = int(x)
        photo_dict["bbox_y"] = int(y)
        photo_dict["bbox_w"] = int(w)
        photo_dict["bbox_h"] = int(h)
        photo_dict["prediction"] = label_pred[i]
        photo_dict["uncertainty"] = round(uncertainty[i], 3)
        for species in strig2num:
            if strig2num[species] < photo_pred.shape[1]:
                photo_dict[species] = round(photo_pred[i, strig2num[species]], 3)
        Table_list.append(photo_dict)
        _, x, y, w, h = new_bounding_boxes[i]
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        space = (int(x+20), int(y+w-40))          

        if num_pred[i] == (photo_pred.shape[1] - 1):
            cv2.rectangle(new_img, pt1, pt2, (255, 0, 0), 15)
        else:
            new_img = cv2.putText(
                img = new_img,
                text = str(get_abbr(label_pred[i])),
                org = space,
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 2,
                color = (0, 0, 0),
                thickness = 3
                )
        
    #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB) #RGB

    if SHOW:
        f,ax = plt.subplots(1,figsize=(8,8))
        ax.imshow(init_img)
        plt.show()

    f,ax = plt.subplots(1,figsize=(8,8))
    ax.imshow(new_img)
    plt.show()

    #cv2.imwrite("./Test/output_image_sample.JPG", cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    output_Table = pd.DataFrame.from_dict(Table_list)

    return output_Table, new_bounding_boxes


def plot_mean_cm(conf_matrices_test, labels, tipo = "Test"):
    mean_conf_matrix_test = np.mean(conf_matrices_test, axis=0)
    std_conf_matrix_test = np.std(conf_matrices_test, axis=0)

    # Create a formatted matrix with accuracy and standard deviation
    formatted_matrix_test = [
        [f"{mean_conf_matrix_test[i, j]:.2f}±{std_conf_matrix_test[i, j]:.2f}"
        for j in range(mean_conf_matrix_test.shape[1])]
        for i in range(mean_conf_matrix_test.shape[0])
    ]

    # Create a confusion matrix display
    cm_display = ConfusionMatrixDisplay(mean_conf_matrix_test, display_labels=labels)

    fig, ax = plt.subplots(figsize=(9, 8))
    cm_display.plot(xticks_rotation=45, cmap = "Grays", include_values = False, ax=ax)#cmap=plt.cm.Blues, ax=ax
    #cm_display.ax_.set_title("")

    # Use text annotation for each cell
    for i in range(mean_conf_matrix_test.shape[0]):
        for j in range(mean_conf_matrix_test.shape[1]):
            if i==j:
                text = ax.text(j, i, str(formatted_matrix_test[i][j]), ha='center', va='center', color = "white")
            else:
                text = ax.text(j, i, str(formatted_matrix_test[i][j]), ha='center', va='center', color = "black")

    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Set axis labels and title
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15)
    ax.set_title(f'Mean {tipo} Confusion Matrix with error', fontsize=18)
    plt.show()


def mean_confmat(label_dict, u_thr, del_unknown=False):
    conf_matrices_val = []
    conf_matrices_test = []
    mean_mic_acc = 0
    mean_mac_acc = 0
    labels = label_dict["species_list"]
    if del_unknown:
        labels = labels[:-1]
    for fold in range(5):
        if fold==4:
            test_species = num2string(label_dict["strig2num"], label_dict["y_true_test"][0])
            conteggio_test = dict(Counter(test_species))

        if label_dict["training_method"] == 'EDL':
            mask_val = label_dict["u_val"][fold]<=u_thr
            mask_test = label_dict["u_test"][fold]<=u_thr
            y_pred_val_fold = label_dict["y_pred_val"][fold][mask_val]
            y_pred_test_fold = label_dict["y_pred_test"][fold][mask_test]
            y_true_val_fold = label_dict["y_true_val"][fold][mask_val]
            y_true_test_fold = label_dict["y_true_test"][fold][mask_test]
        elif label_dict["training_method"] == 'm-EDL':
            y_pred_val_fold = label_dict["y_pred_val"][fold]
            y_pred_test_fold = label_dict["y_pred_test"][fold]
            y_true_val_fold = label_dict["y_true_val"][fold]
            y_true_test_fold = label_dict["y_true_test"][fold]
            if del_unknown:
                target_classes = len(labels)
                mask_val = y_true_val_fold!=target_classes
                mask_test = y_true_test_fold!=target_classes
                y_true_val_fold = y_true_val_fold[mask_val]
                y_true_test_fold = y_true_test_fold[mask_test]
                y_pred_val_fold = y_pred_val_fold[mask_val][:, :-1]
                y_pred_test_fold = y_pred_test_fold[mask_test][:, :-1]

        mic_acc = micro_acc(y_pred_test_fold, y_true_test_fold)
        print(f"The micro accuracy for the fold {fold+1} is: {mic_acc}")
        mean_mic_acc += mic_acc

        mac_acc = macro_acc(y_pred_test_fold, y_true_test_fold)
        print(f"The macro accuracy for the fold {fold+1} is: {mac_acc}")
        mean_mac_acc += mac_acc

        print()

        conf_matrix_val = confusion_matrix(y_true_val_fold, np.argmax(y_pred_val_fold,axis=-1),normalize = "true" )
        conf_matrices_val.append(conf_matrix_val)

        conf_matrix_test = confusion_matrix(y_true_test_fold, np.argmax(y_pred_test_fold,axis=-1),normalize = "true" )
        conf_matrices_test.append(conf_matrix_test)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_val, display_labels=labels)
        disp.plot(xticks_rotation=45,cmap="Grays")
        disp.ax_.set_title(f"Validation confusion matrix for fold {fold+1}")
        plt.show()

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=labels)
        disp.plot(xticks_rotation=45,cmap="Grays")
        disp.ax_.set_title(f"Test confusion matrix for fold {fold+1}")
        plt.show()

    mean_mic_acc = mean_mic_acc/5
    mean_mac_acc = mean_mac_acc/5
    print("The mean micro accuracy over the 5 fold is: ", mean_mic_acc)
    print("The mean macro accuracy over the 5 fold is: ", mean_mac_acc)

    for key, value in conteggio_test.items():
            print(f"{key}: {value}")

    plot_mean_cm(conf_matrices_test, labels, tipo = "Test")
    plot_mean_cm(conf_matrices_val, labels, tipo = "Validation")


# -

def hist_prob_true(label_dict, evidence=False, u_thr=0.8):
    y_prob_test_target = []
    y_prob_test_unknown = []
    for fold in range(5):
        if fold==4:
            test_species = num2string(label_dict["strig2num"], label_dict["y_true_test"][0])
            conteggio_test = dict(Counter(test_species))

        if label_dict["training_method"] == 'EDL':
            mask_val = label_dict["u_val"][fold]<=u_thr
            mask_test = label_dict["u_test"][fold]<=u_thr
            y_pred_val_fold = label_dict["y_pred_val"][fold][mask_val]
            y_pred_test_fold = label_dict["y_pred_test"][fold][mask_test]
            y_true_val_fold = label_dict["y_true_val"][fold][mask_val]
            y_true_test_fold = label_dict["y_true_test"][fold][mask_test]
        elif label_dict["training_method"] == 'm-EDL':
            y_pred_val_fold = label_dict["y_pred_val"][fold]
            y_pred_test_fold = label_dict["y_pred_test"][fold]
            y_true_val_fold = label_dict["y_true_val"][fold]
            y_true_test_fold = label_dict["y_true_test"][fold]

        mask_unknown = y_true_test_fold == (len(label_dict["species_list"]) - 1)
        y_true_test_fold_target = y_true_test_fold[~mask_unknown]
        y_true_test_fold_unknown = y_true_test_fold[mask_unknown]
        y_pred_test_fold_target = y_pred_test_fold[~mask_unknown]
        y_pred_test_fold_unknown = y_pred_test_fold[mask_unknown]

        if evidence:
            strength_target = y_pred_test_fold_target.shape[1] / y_pred_test_fold_target[:, -1]
            y_pred_test_fold_target = (y_pred_test_fold_target * strength_target[:, None]) - 1
            strength_unknown = y_pred_test_fold_unknown.shape[1] / y_pred_test_fold_unknown[:, -1]
            y_pred_test_fold_unknown = (y_pred_test_fold_unknown * strength_unknown[:, None]) - 1


        target_prob = y_pred_test_fold_target[np.arange(y_true_test_fold_target.shape[0]), y_true_test_fold_target]
        #unknown_prob = y_pred_test_fold_unknown[np.arange(y_true_test_fold_unknown.shape[0]), y_true_test_fold_unknown]
        unknown_prob = y_pred_test_fold_unknown[:, -1]

        y_prob_test_target.extend(target_prob)
        y_prob_test_unknown.extend(unknown_prob)

    y_prob_test_target = np.array(y_prob_test_target)
    y_prob_test_unknown = np.array(y_prob_test_unknown)

    text = "evidence" if evidence else "probability"

    num_bins = 15
    min_value = y_prob_test_target.min()
    max_value = y_prob_test_target.max()
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_prob_test_target, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "target prob")
    #ax.hist(y_prob_incorr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect prediction")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel(f'target {text}')
    ax.set_ylabel('% of trials')
    ax.set_title(f"Histograms of target {text} for target species")
    ax.legend()

    num_bins = 15
    min_value = y_prob_test_unknown.min()
    max_value = y_prob_test_unknown.max()
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_prob_test_unknown, bins=bin_edges, density = True, alpha = 0.5, color='red', edgecolor='black', label = "unknown prob")
    #ax.hist(y_prob_incorr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect prediction")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel(f'unknown {text}')
    ax.set_ylabel('% of trials')
    ax.set_title(f"Histograms of unknown {text} for unknown species")
    ax.legend()


def pred_hist(label_dict, u_thr=0.8):
    y_prob_corr_test_target = []
    y_unkown_corr_test_target = []
    y_prob_incorr_test_target = []
    y_unkown_incorr_test_target = []

    y_prob_corr_test_unknown = []
    y_unkown_corr_test_unknown = []
    y_prob_incorr_test_unknown = []
    y_unkown_incorr_test_unknown = []
    for fold in range(5):
        if fold==4:
            test_species = num2string(label_dict["strig2num"], label_dict["y_true_test"][0])
            conteggio_test = dict(Counter(test_species))

        if label_dict["training_method"] == 'EDL':
            mask_val = label_dict["u_val"][fold]<=u_thr
            mask_test = label_dict["u_test"][fold]<=u_thr
            y_pred_val_fold = label_dict["y_pred_val"][fold][mask_val]
            y_pred_test_fold = label_dict["y_pred_test"][fold][mask_test]
            y_true_val_fold = label_dict["y_true_val"][fold][mask_val]
            y_true_test_fold = label_dict["y_true_test"][fold][mask_test]
        elif label_dict["training_method"] == 'm-EDL':
            y_pred_val_fold = label_dict["y_pred_val"][fold]
            y_pred_test_fold = label_dict["y_pred_test"][fold]
            y_true_val_fold = label_dict["y_true_val"][fold]
            y_true_test_fold = label_dict["y_true_test"][fold]

        mask_unknown = y_true_test_fold == (len(label_dict["species_list"]) - 1)
        y_true_test_fold_target = y_true_test_fold[~mask_unknown]
        y_true_test_fold_unknown = y_true_test_fold[mask_unknown]
        y_pred_test_fold_target = y_pred_test_fold[~mask_unknown]
        y_pred_test_fold_unknown = y_pred_test_fold[mask_unknown]

        #y_pred_val = np.argmax(y_pred_val_fold,axis=-1)
        #y_prob_pred_val = np.max(y_pred_val_fold, axis=-1)
        y_pred_test_target = np.argmax(y_pred_test_fold_target, axis=-1)
        y_prob_pred_test_target = np.max(y_pred_test_fold_target, axis=-1)
        y_pred_test_unknown = np.argmax(y_pred_test_fold_unknown, axis=-1)
        y_prob_pred_test_unknown = np.max(y_pred_test_fold_unknown, axis=-1)
        #print(y_prob_pred_test[y_prob_pred_test>1])

        #mask_correct_val = y_pred_val == y_true_val_fold
        #y_prob_corr_val = y_prob_pred_val[mask_correct_val]
        #y_prob_incorr_val = y_prob_pred_val[~mask_correct_val]
        mask_correct_test_unknown = y_pred_test_unknown == y_true_test_fold_unknown
        mask_correct_test_target = y_pred_test_target == y_true_test_fold_target
        y_prob_corr_test_target.extend(y_prob_pred_test_target[mask_correct_test_target])
        y_unkown_corr_test_target.extend(y_pred_test_fold_target[mask_correct_test_target, -1])
        y_prob_incorr_test_target.extend(y_prob_pred_test_target[~mask_correct_test_target])
        y_unkown_incorr_test_target.extend(y_pred_test_fold_target[~mask_correct_test_target, -1])

        y_prob_corr_test_unknown.extend(y_prob_pred_test_unknown[mask_correct_test_unknown])
        y_unkown_corr_test_unknown.extend(y_pred_test_fold_unknown[mask_correct_test_unknown, -1])
        y_prob_incorr_test_unknown.extend(y_prob_pred_test_unknown[~mask_correct_test_unknown])
        y_unkown_incorr_test_unknown.extend(y_pred_test_fold_unknown[~mask_correct_test_unknown, -1])

    y_prob_corr_test_target = np.array(y_prob_corr_test_target)
    y_unkown_corr_test_target = np.array(y_unkown_corr_test_target)
    y_prob_incorr_test_target = np.array(y_prob_incorr_test_target)
    y_unkown_incorr_test_target = np.array(y_unkown_incorr_test_target)

    y_prob_corr_test_unknown = np.array(y_prob_corr_test_unknown)
    y_unkown_corr_test_unknown = np.array(y_unkown_corr_test_unknown)
    y_prob_incorr_test_unknown = np.array(y_prob_incorr_test_unknown)
    y_unkown_incorr_test_unknown = np.array(y_unkown_incorr_test_unknown)

    num_bins = 15
    min_value = min(y_prob_corr_test_target.min(), y_prob_incorr_test_target.min())
    max_value = max(y_prob_corr_test_target.max(), y_prob_incorr_test_target.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_prob_corr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='red', edgecolor='black', label = "correct prediction")
    ax.hist(y_prob_incorr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect prediction")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel('prediction probability')
    ax.set_ylabel('% of trials')
    ax.set_title("Histograms of prediction probability for correct and incorrect predictions of target species")
    ax.legend()


    num_bins = 15
    min_value = min(y_prob_corr_test_unknown.min(), y_prob_incorr_test_unknown.min())
    max_value = max(y_prob_corr_test_unknown.max(), y_prob_incorr_test_unknown.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_prob_corr_test_unknown, bins=bin_edges, density = True, alpha = 0.5, color='red', edgecolor='black', label = "correct prediction")
    ax.hist(y_prob_incorr_test_unknown, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect prediction")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel('prediction probability')
    ax.set_ylabel('% of trials')
    ax.set_title("Histograms of prediction probability for correct and incorrect predictions of unknown species")
    ax.legend()


    min_value = min(y_unkown_corr_test_target.min(), y_unkown_incorr_test_target.min())
    max_value = max(y_unkown_corr_test_target.max(), y_unkown_incorr_test_target.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_unkown_corr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='red', edgecolor='black', label = "correct unknown prob")
    ax.hist(y_unkown_incorr_test_target, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect unknown prob")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel('unknown probability')
    ax.set_ylabel('% of trials')
    ax.set_title("Histograms of unknown probability for correct and incorrect predictions of target species")
    ax.legend()


    min_value = min(y_unkown_corr_test_unknown.min(), y_unkown_incorr_test_unknown.min())
    max_value = max(y_unkown_corr_test_unknown.max(), y_unkown_incorr_test_unknown.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, ax = plt.subplots()
    ax.hist(y_unkown_corr_test_unknown, bins=bin_edges, density = True, alpha = 0.5, color='red', edgecolor='black', label = "correct unknown prob")
    ax.hist(y_unkown_incorr_test_unknown, bins=bin_edges, density = True, alpha = 0.5, color='skyblue', edgecolor='black', label = "incorrect unknown prob")
    #ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    #ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel('unknown probability')
    ax.set_ylabel('% of trials')
    ax.set_title("Histograms of unknown probability for correct and incorrect predictions of unknown species")
    ax.legend()


# +
def accuracy_vs_uthr(label_dict, points):
    n_target_classes = len(label_dict["species_list"])
    conf_matrices_val = []
    conf_matrices_test = []
    mean_mic_acc_u = []
    mean_mac_acc_u = []
    u_thr_eff_list = []
    points = 10
    u_thr_list = np.linspace(0.1, 1, points)
    diagonals = []
    mean_frac = []
    mean_micro = []
    for q, u_thr in enumerate(u_thr_list):
        mean_mic_acc = 0
        mean_mac_acc = 0
        micro_frac = 0
        tot_frac = np.zeros(n_target_classes)
        conf_matrices_test = []
        invalid_fold = False
        for fold in range(5):
            if fold==4:
                test_species = num2string(label_dict["strig2num"], label_dict["y_true_test"][0])
                conteggio_test = dict(Counter(test_species))


            mask_test = label_dict["u_test"][fold]<=u_thr
            y_pred_test_fold = label_dict["y_pred_test"][fold][mask_test]
            y_true_test_fold = label_dict["y_true_test"][fold][mask_test]

            tot_test = len(label_dict["y_true_test"][fold])
            tot_true_test = np.array([(label_dict["y_true_test"][fold] == i).sum() for i in range(n_target_classes)])
            tot_mask = len(y_true_test_fold)
            tot_mask_test = np.array([(y_true_test_fold == i).sum() for i in range(n_target_classes)])
            frac_unknown = (tot_true_test - tot_mask_test) / tot_true_test
            frac_tot = (tot_test - tot_mask) / tot_test

            tot_frac += frac_unknown
            micro_frac += frac_tot

            mic_acc = micro_acc(y_pred_test_fold, y_true_test_fold)
            mean_mic_acc += mic_acc

            mac_acc = macro_acc(y_pred_test_fold, y_true_test_fold)
            mean_mac_acc += mac_acc

            conf_matrix_test = confusion_matrix(y_true_test_fold, np.argmax(y_pred_test_fold,axis=-1),normalize = "true" )
            if conf_matrix_test.shape != (n_target_classes, n_target_classes):
                # se la shape non è corretta, setta il flag e interrompi il ciclo
                invalid_fold = True
                break 

            conf_matrices_test.append(conf_matrix_test)

        if invalid_fold:
            # Se è successo un problema, passa direttamente al prossimo u_thr
            continue

        print(u_thr)
        u_thr_eff_list.append(u_thr)
        mean_frac.append(tot_frac/5)
        mean_micro.append(micro_frac/5)
        mean_mic_acc_u.append(mean_mic_acc/5) 
        mean_mac_acc_u.append(mean_mac_acc/5)

        conf_matrices_test = np.array(conf_matrices_test)
        mean_conf_matrix_test = np.mean(conf_matrices_test, axis=0)

        diagonale = np.array([mean_conf_matrix_test[i][i] for i in range(len(mean_conf_matrix_test))])
        diagonals.append(diagonale) 

    diagonals = np.array(diagonals)
    mean_frac = np.array(mean_frac)
    mean_micro = np.array(mean_micro)
    mean_macro = mean_frac.mean(1)
    colors = [
        "#1f77b4",  # blu
        "#ff7f0e",  # arancione
        "#2ca02c",  # verde
        "#d62728",  # rosso
        "#9467bd",  # viola
        "#8c564b",  # marrone
        "#e377c2"   # rosa
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    #ax2 = ax.twinx()
    #ax2.set_ylabel("% of labelled as unknown")
    #ax2.set_ylim((0, 1))
    for species in range(n_target_classes):
        ax.plot(u_thr_eff_list, diagonals[:, species], c=colors[species], label = label_dict["species_list"][species])
        ax.plot(u_thr_eff_list, mean_frac[:, species], '--', c=colors[species], label = label_dict["species_list"][species])
    ax.set_xlabel('Uncertainty_thr')
    ax.set_ylabel('species accuracy')
    #ax.set_title("Histograms of Uncertainty for target and unknown species")
    ax.legend()

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(u_thr_eff_list, mean_mic_acc_u, color='r', label = "micro accuracy media")
    ax.plot(u_thr_eff_list, mean_mac_acc_u, color='b', label = "macro accuracy media")
    ax.plot(u_thr_eff_list, mean_micro, '--', color='r', label = "micro fraction scartata")
    ax.plot(u_thr_eff_list, mean_macro, '--', color='b', label = "macro fraction scartata")
    ax.set_xlabel('Uncertainty_thr')
    ax.set_ylabel('accuracy')
    #ax.set_title("Histograms of Uncertainty for target and unknown species")
    ax.legend()

    incertezza_pred = np.concatenate((label_dict["u_test"][fold], label_dict["u_unknown"][fold]), axis=0)
    incertezza_compl = 1-incertezza_pred
    #u_pred = np.concatenate((incertezza_compl, incertezza_pred), axis=1)
    u_true_test = np.zeros(label_dict["u_test"][fold].shape[0], dtype = int) # classe 0 target
    u_true_unknown = np.ones(label_dict["u_unknown"][fold].shape[0], dtype = int) # classe 1 unknown
    u_true = np.concatenate((u_true_test, u_true_unknown), axis=0)
    u_pred = np.zeros((u_true.shape[0], 2))
    u_pred[:, 0] = incertezza_compl
    u_pred[:, 1] = incertezza_pred
    #print(u_true)
    #print(u_pred)


def acc_vs_uthr_EDL(label_dict, fold, u_thr_list):
    n_target_classes = len(label_dict["species_list"])
    n_tot_classes = len(label_dict["strig2num"])
    num2strig = {value: key for key, value in label_dict["strig2num"].items()}

    y_true = np.concatenate((label_dict["y_true_test"][fold], label_dict["y_true_unknown"][fold]), axis=0)
    y_pred_prob = np.concatenate((label_dict["y_pred_test"][fold], label_dict["y_pred_unknown"][fold]), axis=0)
    uncertainties = np.concatenate((label_dict["u_test"][fold], label_dict["u_unknown"][fold]), axis=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.where(y_true >= n_target_classes, n_target_classes, y_true)
    micro_acc = np.zeros(len(u_thr_list))
    macro_acc = np.zeros(len(u_thr_list))
    for q, u_thr in enumerate(u_thr_list):
        # Costruzione delle predizioni modificate
        y_pred_with_unknowns = np.where(uncertainties > u_thr, n_target_classes, y_pred)
        micro_acc[q] = ((y_pred_with_unknowns == y_true).sum())/y_true.shape[0]
        # Costruzione di una matrice 14 x 7: righe = vere classi, colonne = 6 note + 1 unknown
        cm_extended = np.zeros(((n_target_classes + 1), (n_target_classes + 1)), dtype=int)
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_label = y_pred_with_unknowns[i]
            cm_extended[true_label, pred_label] += 1

        # Normalizzazione riga per riga (somma = 100%)
        cm_normalized = cm_extended.astype(np.float64)
        row_sums = cm_normalized.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm_normalized, row_sums, where=row_sums != 0) * 100

        print(cm_percent.shape)
        macro_acc_tot = [cm_percent[i][i] for i in range(cm_percent.shape[0])]
        macro_acc[q] = macro_acc_tot.sum()/cm_percent.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(u_thr_list, micro_acc, color='r', label = "micro accuracy")
    ax.plot(u_thr_list, macro_acc, color='b', label = "macro accuracy")
    ax.set_xlabel('Uncertainty_thr')
    ax.set_ylabel('accuracy')
    #ax.set_title("Histograms of Uncertainty for target and unknown species")
    ax.legend()


# +
def general_cm_EDL(label_dict, fold, u_thr=0.25):
    n_target_classes = len(label_dict["species_list"])
    n_tot_classes = len(label_dict["strig2num"])
    num2strig = {value: key for key, value in label_dict["strig2num"].items()}

    y_true = np.concatenate((label_dict["y_true_test"][fold], label_dict["y_true_unknown"][fold]), axis=0)
    y_pred_prob = np.concatenate((label_dict["y_pred_test"][fold], label_dict["y_pred_unknown"][fold]), axis=0)
    uncertainties = np.concatenate((label_dict["u_test"][fold], label_dict["u_unknown"][fold]), axis=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Costruzione delle predizioni modificate
    y_pred_with_unknowns = np.where(uncertainties > u_thr, n_target_classes, y_pred)
    #y_true = np.where(y_true >= n_classi, n_classi, y_true)

    # Costruzione di una matrice 14 x 7: righe = vere classi, colonne = 6 note + 1 unknown
    cm_extended = np.zeros((n_tot_classes, (n_target_classes + 1)), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred_with_unknowns[i]
        cm_extended[true_label, pred_label] += 1

    # Normalizzazione riga per riga (somma = 100%)
    cm_normalized = cm_extended.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_normalized, row_sums, where=row_sums != 0) * 100

    # Etichette
    true_labels_names = [num2strig[i] for i in range(n_tot_classes)]
    pred_labels_names = [num2strig[i] for i in range(n_target_classes)] + ['Unknown']

    # Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=pred_labels_names,
                yticklabels=true_labels_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix con Classe Sconosciuta, uncert_thr = {u_thr} ({n_target_classes} classi)')
    plt.tight_layout()
    plt.show()


def conf_with_unknown_EDL(label_dict, fold, u_thr=0.25):
    n_target_classes = len(label_dict["species_list"])
    n_tot_classes = len(label_dict["strig2num"])
    num2strig = {value: key for key, value in label_dict["strig2num"].items()}

    y_true = np.concatenate((label_dict["y_true_test"][fold], label_dict["y_true_unknown"][fold]), axis=0)
    y_pred_prob = np.concatenate((label_dict["y_pred_test"][fold], label_dict["y_pred_unknown"][fold]), axis=0)
    uncertainties = np.concatenate((label_dict["u_test"][fold], label_dict["u_unknown"][fold]), axis=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Costruzione delle predizioni modificate
    y_pred_with_unknowns = np.where(uncertainties > u_thr, n_target_classes, y_pred)
    y_true = np.where(y_true >= n_target_classes, n_target_classes, y_true)
    micro_acc = ((y_pred_with_unknowns == y_true).sum())/y_true.shape[0]

    # Costruzione di una matrice 14 x 7: righe = vere classi, colonne = 6 note + 1 unknown
    cm_extended = np.zeros(((n_target_classes + 1), (n_target_classes + 1)), dtype=int)

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred_with_unknowns[i]
        cm_extended[true_label, pred_label] += 1

    # Normalizzazione riga per riga (somma = 100%)
    cm_normalized = cm_extended.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_normalized, row_sums, where=row_sums != 0) * 100

    # Etichette
    true_labels_names = [num2strig[i] for i in range(n_target_classes)] + ['Unknown']
    pred_labels_names = [num2strig[i] for i in range(n_target_classes)] + ['Unknown']

    print(cm_percent.shape)
    macro_acc_tot = [cm_percent[i][i] for i in range(cm_percent.shape[0])]
    macro_acc = macro_acc_tot.sum()/cm_percent.shape[0]

    print(f"The micro accuracy of the fold {fold}, with u_thr={u_thr}, is: {micro_acc}")
    print(f"The macro accuracy of the fold {fold}, with u_thr={u_thr}, is: {macro_acc}")

    # Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=pred_labels_names,
                yticklabels=true_labels_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix con Classe Sconosciuta, uncert_thr = {u_thr} ({n_target_classes} classi)')
    plt.tight_layout()
    plt.show()




def general_cm_mEDL(label_dict, y_true_test, y_pred_test):
    n_target_classes = len(label_dict["species_list"])
    n_tot_classes = len(label_dict["strig2num"])
    y_pred = np.argmax(y_pred_test, axis=1)
    micro_acc = ((y_pred == y_true_test).sum())/y_true_test.shape[0]
    # Costruzione di una matrice 14 x 7: righe = vere classi, colonne = 6 note + 1 unknown
    cm_extended = np.zeros((n_tot_classes, n_target_classes ), dtype=int)
    for i in range(len(y_true_test)):
        true_label = y_true_test[i]
        pred_label = y_pred[i]
        cm_extended[true_label, pred_label] += 1

    # Normalizzazione riga per riga (somma = 100%)
    cm_normalized = cm_extended.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_normalized, row_sums, where=row_sums != 0) * 100

    # Etichette
    true_labels_names = list(label_dict["strig2num"].keys())
    pred_labels_names = label_dict["species_list"]
    # Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=pred_labels_names,
                yticklabels=true_labels_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix con Classe Sconosciuta ({n_target_classes} classi)')
    plt.tight_layout()
    plt.show()


def hist_species_uncert(label_dict, fold):
    n_target_classes = len(label_dict["species_list"])
    n_tot_classes = len(label_dict["strig2num"])
    num2strig = {value: key for key, value in label_dict["strig2num"].items()}


    y_chosen_test = np.argmax(label_dict["y_pred_test"][fold], axis=1)

    for classe in range(n_target_classes):
        mask_classe = label_dict["y_true_test"][fold]==classe
        y_true_classe = label_dict["y_true_test"][fold][mask_classe]
        y_chosen_classe = y_chosen_test[mask_classe]
        incertezza_classe = label_dict["u_test"][fold][mask_classe]

        mask_right = y_chosen_classe==y_true_classe
        mask_wrong = y_chosen_classe!=y_true_classe
        incertezza_right = incertezza_classe[mask_right]
        incertezza_wrong = incertezza_classe[mask_wrong]#[~mask_right]

        num_bins = 15
        min_value = min(incertezza_right.min(), incertezza_wrong.min())
        max_value = max(incertezza_right.max(), incertezza_wrong.max())
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)


        fig, ax = plt.subplots()
        ax.hist(incertezza_right, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "right answers")
        ax.hist(incertezza_wrong, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "wrong answers")
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('% of trials')
        ax.set_title(f"Uncertainty Histograms for right and wrong answers for {num2strig[classe]}")
        ax.legend()


def target_vs_unknown(incertezza_test, incertezza_unknown, x_lims, n): 
    l_lim, r_lim = x_lims 
    u_thr_list = np.linspace(l_lim, r_lim, n)
    l_test = incertezza_test.shape[0]
    l_unknown = incertezza_unknown.shape[0]
    perc_test = np.zeros(n)
    perc_unknown = np.zeros(n)
    for i, u_thr in enumerate(u_thr_list):
        l_test_thr = (incertezza_test < u_thr).sum() 
        l_unknown_thr = (incertezza_unknown > u_thr).sum() 
        perc_test[i] = l_test_thr/l_test
        perc_unknown[i] = l_unknown_thr/l_unknown

    diff = perc_test - perc_unknown

    # Cerchiamo l'indice in cui il segno di diff cambia (cioè, diff[i] * diff[i+1] < 0)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # C'è un cambio di segno tra t[i] e t[i+1]
            # Usiamo l'interpolazione lineare per stimare l'istante dell'intersezione:
            # Formula: t_int = t[i] + (t[i+1] - t[i]) * (|diff[i]| / (|diff[i]| + |diff[i+1]|))
            t0, t1 = u_thr_list[i], u_thr_list[i+1]
            d0, d1 = diff[i], diff[i+1]
            intersection_uncertainty = t0 + (t1 - t0) * (abs(d0) / (abs(d0) + abs(d1)))    

    l_test_final = (incertezza_test < intersection_uncertainty).sum() 
    l_unknown_final = (incertezza_unknown > intersection_uncertainty).sum() 
    perc_test_final = l_test_final/l_test
    perc_unknown_final = l_unknown_final/l_unknown


    print(f"Percentage of target species recognized as target with \
    uncertainty_thr={intersection_uncertainty:.2f}: {perc_test_final*100:.1f}%")

    print(f"Percentage of unknown species recognized as unknown with \
    uncertainty_thr={intersection_uncertainty:.2f}: {perc_unknown_final*100:.1f}%")    

    fig, ax = plt.subplots()
    ax.plot(u_thr_list, perc_test, color='red', label = "target species")
    ax.plot(u_thr_list, perc_unknown, color='b', label = "unknown species")
    plt.axvline(x=intersection_uncertainty, color='black', linestyle='--', label=f'Intersezione a u_thr={intersection_uncertainty:.2f}')
    ax.set_xlabel('Uncertainty_thr')
    ax.set_ylabel('Percentage')
    #ax.set_title("Histograms of Uncertainty for target and unknown species")
    ax.legend()


    num_bins = 30
    min_value = min(incertezza_test.min(), incertezza_unknown.min())
    max_value = max(incertezza_test.max(), incertezza_unknown.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    print("Incertezza media ")

    fig, ax = plt.subplots()
    ax.hist(incertezza_test, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "target species")
    ax.hist(incertezza_unknown, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "unknown species")
    ax.axvline(incertezza_test.mean(), color="r",linestyle="--",label="mean_uncertainty on target")
    ax.axvline(incertezza_unknown.mean(), color="skyblue",linestyle="--",label="mean_uncertainty on unknown")
    # Add labels and title
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of Uncertainty for target and unknown species")
    ax.legend()

