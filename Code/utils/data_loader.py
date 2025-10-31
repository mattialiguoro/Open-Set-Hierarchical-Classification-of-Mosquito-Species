from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# FOR TOT_FIELD_DATASET
class BoundingBoxTransformer:
    def __init__(self, debug=False):
        self.debug = debug
        self.positions = ['green_new', 'gray_new', 'gray_up', 'gray_down', 'green_up_1', 'green_up_2', 'green_down_1', 'green_down_2']

        self.grid_image_paths = {
            "green_new": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_green_new.JPG",
            "gray_new": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_gray_new.JPG",
            "gray_up": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_none_up.JPG",
            "gray_down": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_none_down.JPG",
            "green_up_1": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_su_1.JPG",
            "green_up_2": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_su_2.JPG",
            "green_down_1": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_giu_1.JPG",
            "green_down_2": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_giu_2.JPG"
        }
        self.annotation_paths = {
            "green_new": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_green_new.txt",
            "gray_new": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_gray_new.txt",
            "gray_up": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_none_up.txt",
            "gray_down": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_none_down.txt",
            "green_up_1": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_su_1.txt",
            "green_up_2": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_su_2.txt",
            "green_down_1": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_giu_1.txt",
            "green_down_2": Path(__file__).resolve().parent.parent / "MOSAICOgrid" / "mosaico_verde_giu_2.txt"
        }
        self.grid_color = [60, 60, 60]#[94, 170, 17], [110, 180, 15]  [60, 60, 60]# Example color of the grid
        self.tolerance = [55, 15, 6]#[55, 20, 12], [60, 35, 8] [55, 15, 6] # Tolerance for color selection
        self.erosion_kernel = np.ones((10, 10), np.uint8)

        #grid_color = [94,170,17]#rgb --> bgr
        #tolerance = [55, 20, 12]  # Tolerance for HSV channels

        # Inizializza dizionari per memorizzare immagini e annotazioni
        self.grid_images = {}
        self.annotations = {}
        self.bounding_boxes = {}
        self.warp_matrix = {}

        # Carica immagini e annotazioni per entrambe le posizioni
        for pos in self.positions:
            self.grid_images[pos], self.annotations[pos] = self.load_image_and_annotations(
                self.grid_image_paths[pos],
                self.annotation_paths[pos]
            )
            self.bounding_boxes[pos] = self.yolo_to_bounding_boxes(
                self.annotations[pos],
                self.grid_images[pos].shape
            )
            self.warp_matrix[pos] = np.eye(3, dtype=np.float32)
        self.log("Grid and annotations loaded successfully.")

    def log(self, message):
        if self.debug:
            print(f"[LOG]: {message}")

    def load_image_and_annotations(self, image_path, annotation_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        annotations = []
        if annotation_path:
            with annotation_path.open('r') as f:
                annotations = f.readlines()

        return image, annotations

    def yolo_to_bounding_boxes(self, annotations, image_shape): # from annotations to bounding_boxes
        bounding_boxes = []
        height, width, _ = image_shape

        w_list = []
        h_list = []
        for annotation in annotations:
            values = annotation.strip().split()
            if len(values) != 5:
                continue
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, values)
            x = int((x_center - bbox_width / 2) * width)
            y = int((y_center - bbox_height / 2) * height)
            w = int(bbox_width * width)
            h = int(bbox_height * height)
            w_list.append(w)
            h_list.append(h)
            bounding_boxes.append((class_id, x, y, w, h))

        self.w_max = max(w_list)
        self.h_max = max(h_list)
        return bounding_boxes

    def select_similar_color(self, image):

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        target_color = cv2.cvtColor(np.uint8([[self.grid_color]]), cv2.COLOR_BGR2Lab)[0][0]

        lower_bound = np.maximum(0, target_color - self.tolerance)
        upper_bound = np.minimum([255, 255, 255], target_color + self.tolerance)#, dtype=np.uint8)

        mask = cv2.inRange(lab_image, lower_bound, upper_bound)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def compute_transformation(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        mask_image = self.select_similar_color(image)
        moving_image = cv2.normalize(mask_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        moving_image = cv2.resize(moving_image, (600, 400))

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
        cc = {}
        mask_grid = {}
        fixed_image = {}
        warp_matrix = {}
        best_cc = 0
        for pos in self.positions:
            mask_grid[pos] = self.select_similar_color(self.grid_images[pos])
            fixed_image[pos] = cv2.normalize(mask_grid[pos], None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
            fixed_image[pos] = cv2.resize(fixed_image[pos], (600, 400))
            fixed_image[pos][360:, :] = 0
            # Assicurati che warp_matrix sia float32 e nel formato corretto
            warp_matrix[pos] = np.eye(3, dtype=np.float32)

            try:
                cc[pos], warp_matrix[pos] = cv2.findTransformECC(
                    fixed_image[pos],
                    moving_image,
                    warp_matrix[pos],
                    cv2.MOTION_HOMOGRAPHY,
                    criteria
                )
                self.warp_matrix[pos] = warp_matrix[pos]
                #self.log(f"Alignment successful! Correlation Coefficient: {cc[pos]}")
            except cv2.error as e:
                #self.log(f"Error in alignment for position {pos}: {str(e)}")
                # In caso di errore, mantieni la matrice identità
                cc[pos] = 0
                warp_matrix[pos] = np.eye(3, dtype=np.float32)

            if cc[pos] > best_cc:
                    best_cc = cc[pos]
                    best_pos = pos

        #print(f"Best cc: {best_cc}")
        #print(f"Best pos: {best_pos}")
        self.best_pos = best_pos
        self.best_cc = best_cc

        # Scaling matrix
        s_x = mask_grid[self.best_pos].shape[1]/600
        s_y = mask_grid[self.best_pos].shape[0]/400
        S = np.array([[s_x, 0, 0],
                      [0, s_y, 0],
                      [0, 0, 1]], dtype=np.float32)

        # Adjust homography matrix
        self.warp_matrix[self.best_pos] = S @ self.warp_matrix[self.best_pos] @ np.linalg.inv(S)
        self.warp_matrix[self.best_pos] = self.warp_matrix[self.best_pos].astype(np.float32)  # Assicurati che sia float32
        #plt.imshow(mask_grid[self.best_pos])
        #plt.show()
        #plt.imshow(mask_image)
        #plt.show()

    def transform_bounding_boxes(self, image_path, label_dict, SHOW=False, margin=62):
        self.compute_transformation(image_path)
        log_bb = {}
        if self.best_cc < 0.3:
            log_bb["status"] = False
            log_bb["message"] = "Image not processed, because does not match with any of the possible grids."
            self.log("Correlation Coefficient is too low. Bounding boxes not transformed.")
            return [], [], [], [], log_bb

        if self.best_pos == "gray_down":
            self.bounding_boxes[self.best_pos] = self.bounding_boxes[self.best_pos][:-1]
        vertexs = [self.bbox2vert(b) for b in self.bounding_boxes[self.best_pos]]
        coordinates, classes = self.list_coordinates_vertex(vertexs)
        transformed_coordinates = cv2.perspectiveTransform(coordinates, self.warp_matrix[self.best_pos])
        transformed_vertexs = self.back_to_vertex(transformed_coordinates, classes)
        transformed_bounding_boxes = [self.vert2bbox(v) for v in transformed_vertexs]
        new_bounding_boxes = []
        mosquito_array = []
        label_array = []
        img = cv2.imread(image_path) # cv2 legge in BGR
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB
        group_img = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # GRAY
        count=0
        for i, box in enumerate(transformed_bounding_boxes):
            class_id, x, y, w, h = box
            if class_id==0:
                height, width = gray_img.shape
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                x, y, w, h = map(int, (x, y, w, h))

                # Controlla se la ROI ha dimensioni valide
                if w <= 0 or h <= 0:
                    self.log(f"Invalid box dimensions at index {i}: w={w}, h={h}")
                    continue
                # Estrae la ROI (Region of Interest)
                roi = gray_img[y:y+h, x:x+w] #GRAY
                if roi.shape[0] <= 2*margin or roi.shape[1] <= 2*margin:
                    # Usa un margine più piccolo o nessun margine
                    adjusted_margin = min(margin, roi.shape[0]//4, roi.shape[1]//4)
                    self.log(f"Adjusting margin from {margin} to {adjusted_margin} for small ROI")
                    roi_inner = roi[adjusted_margin:-adjusted_margin, adjusted_margin:-adjusted_margin] if adjusted_margin > 0 else roi
                else:
                    roi_inner = roi[margin:-margin, margin:-margin] #GRAY

                # Verifica che roi_inner non sia vuoto
                if roi_inner.size == 0:
                    self.log(f"Empty ROI at index {i}")
                    continue

                try:
                    th1 = cv2.adaptiveThreshold(roi_inner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,257,25)
                    #kernel = np.ones((10, 10), np.uint8)
                    eroded = cv2.erode(th1, self.erosion_kernel,iterations=2)
                    #dilate_kernel = np.ones((5, 5), np.uint8)  # non attivo prima dello smazzamento della chiamata
                    dilated = cv2.dilate(eroded, self.erosion_kernel,iterations=10)

                    # Analisi dei componenti connessi
                    analysis = cv2.connectedComponentsWithStats(dilated, 4, cv2.CV_32S)
                    (totalLabels, _, _, _) = analysis

                    # Modifica la condizione per il rilevamento della zanzara
                    if totalLabels >= 2:  # Cambiato da == 2 a >= 2 per essere più permissivo
                        count+=1
                        new_bounding_boxes.append(box)
                        if y+self.h_max > new_img.shape[0] or x+self.w_max > new_img.shape[1]:
                            log_bb["status"] = False
                            log_bb["message"] = "Image not processed, because the grid extends beyond the image boundaries."
                            self.log("Error: attempting to create a bounding box which goes out of the image")
                            return [], [], [], [], log_bb
                        cutted_mosquito = new_img[y:y+self.h_max, x:x+self.w_max,:] #RGB
                        mosquito_array.append(cutted_mosquito)
                        label_dict_copy = label_dict.copy()
                        label_dict_copy["photo_index"] = count
                        label_dict_copy["bbox_x"] = x
                        label_dict_copy["bbox_y"] = y
                        label_dict_copy["bbox_w"] = w
                        label_dict_copy["bbox_h"] = h
                        label_array.append(label_dict_copy)
                except cv2.error as e:
                    self.log(f"OpenCV error processing box {i}: {str(e)}")
                    continue

        n_mosquito = len(mosquito_array)
        for diz in label_array:
            diz["n_mosquito_found"] = n_mosquito
        for i, box in enumerate(new_bounding_boxes):
            class_id, x, y, w, h = box
            x, y, w, h = map(int, (x, y, w, h))
            cv2.rectangle(group_img, (x, y), (x + w, y + h), (0, 255, 0), 15) #BGR
            #cv2.putText(group_img, f"{int(i+1)}", (x+20, y+h-40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        if SHOW:
            plt.imshow(group_img)
            plt.show()
        log_bb["status"] = True
        log_bb["message"] = "Image processed successfully. Results now has to be saved in the database."
        return np.array(new_bounding_boxes), np.array(mosquito_array), np.array(label_array), group_img, log_bb

    def back_to_vertex(self, coordinates, classes):
        vertexs = []
        for i in range(len(classes)):
            co = coordinates[4*i:4*(i+1), 0, :].astype("int")
            if co.shape != (4, 2):
                raise ValueError(f"Invalid transformed vertex shape: {co.shape}")
            c = classes[i]
            vertexs.append([c, tuple(co[0]), tuple(co[1]), tuple(co[2]), tuple(co[3])])
        return vertexs


    def bbox2vert(self, bbox):
        c, x, y, w, h = bbox
        return [c, (x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    def vert2bbox(self, vertices):
        c = vertices[0]
        vertices = np.array(vertices[1:], dtype=np.float32)
        x_min, y_min = vertices[:, 0].min(), vertices[:, 1].min()
        x_max, y_max = vertices[:, 0].max(), vertices[:, 1].max()
        return (c, int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def list_coordinates_vertex(self, vertexs):
        # Convert a list of vertices into an array formatted for perspective transformation.
        coordinates = np.zeros((len(vertexs) * 4, 1, 2), dtype=np.float32)
        classes = np.zeros(len(vertexs))
        for i, v in enumerate(vertexs):
            classes[i] = v[0]
            points = np.array(v[1:], dtype=np.float32).reshape(4, 2)  # Ensure shape is (4,2)
            coordinates[4 * i:4 * (i + 1), 0, :] = points
        return coordinates, classes


# FOR TOT_FIELD_DATASET
def get_data_listed(dataset):
    root_path = Path(__file__).resolve().parent.parent
    data_path = root_path / dataset / "Original_photo"
    table_path = root_path / dataset / "Original_Table.xlsx"
    Table = pd.read_excel(table_path)#, sheet_name=0)
    images_list = []  # lista di path relativi a ciascun file .JPG
    label_list = []  # lista di labels (dizionari)
    ID_list = []   # lista di ID (stringhe)
    for file in Path(data_path).iterdir():
        #print(f"file n.{i+1}: {file}")
        if file.endswith(".JPG"):
            label = {}
            file_name = ".".join(file.split(".")[:-1])
            print(file_name)
            #source_tmp = file_name.split("_")[1]
            #numeric_index = ''.join(i for i in source_tmp if i.isdigit())
            mask_file = Table["ID_FOTO"]==file_name
            species = Table[mask_file]["SPECIE"].values[0]
            source = Table[mask_file]["AREA GEOGRAFICA"].values[0]
            origin = Table[mask_file]["ORIGINE"].values[0]
            n_mosquito = Table[mask_file]["N° esemplari per foto"].values[0]
            #ID = "_".join((file.split("_")[0], source, numeric_index, origin))
            new_path = root_path / dataset / "Group_photo" / file
        
            label["file"]= file[:-4]
            label["path"]= new_path
            label["species"] = species
            label["n_mosquito"] = n_mosquito
            label["source"] = source
            label["origin"] = origin
            #label["source_index"] = numeric_index

            images_list.append(Path(data_path) / file)
            ID_list.append(file[:-4])
            label_list.append(label)
    return images_list, label_list, ID_list






# GENERAL FUNCITONS
def split_data_by_source(path, species_list, test_sources=""):
    Table = pd.read_excel(path, sheet_name=0)
    # groupby() raggruppa insieme tutte le righe con la stessa specie.
    # agg() dice di scrivere le sources senza ripetizioni e di scrivere i file separati da una virgola
    source_summary = Table.groupby("species", as_index=False).agg({"source": np.unique, "file": ", ".join})
    train_files, test_files = [], []
    report_train, report_test = {}, {}
    for s in species_list:
        sources = source_summary[source_summary["species"] == s]["source"].values[0]
        l = len(sources)
        # se una specie ha solo una source finisce solo nel train_set e non sarà presente nel test_set
        if l == 1:
            filther_train = (Table.species==s) & (Table.source==sources[0])
            n_mosquito_train = Table[filther_train]['n_mosquito'].sum()
            report_train[s] = ([sources[0]], n_mosquito_train)
            selected_files_train = Table[filther_train]["file"].apply(lambda x: "{}{}".format(x,".npz")).values
            # Seleziono i file relativi a specie s e source su, e li converto in npz, per poi metterli in train_files.
            # selected_files = Table.query("species == @s and source == @su")["file"].apply(lambda x: "{}{}".format(x,".npz")).values
            train_files.extend(selected_files_train)
        else:
            if test_sources:
                test = np.array([any(a == b for b in test_sources[s]) for a in sources])
            else:
                test_size = l//3
                # Scelgo l/3 (arrotondato per difetto) sources a caso tra quelle della specie s, e le uso nel test_set.
                # Le restanti le uso nel train_set
                choice = np.random.choice(range(l), size=test_size, replace=False)
                test = np.zeros(sources.shape[0], dtype=bool)
                test[choice] = True
            train = ~test
            filther_train = (Table.species==s) & Table["source"].isin(sources[train])
            filther_test = (Table.species==s) & Table["source"].isin(sources[test])
            n_mosquito_train = Table[filther_train]['n_mosquito'].sum()
            n_mosquito_test = Table[filther_test]['n_mosquito'].sum()
            report_train[s] = (list(sources[train]), n_mosquito_train)
            report_test[s] = (list(sources[test]), n_mosquito_test)
            #print(sources[train])
            #print(sources[test])
            selected_files_train = Table[filther_train]["file"].apply(lambda x: "{}{}".format(x,".npz")).values
            selected_files_test = Table[filther_test]["file"].apply(lambda x: "{}{}".format(x,".npz")).values
            train_files.extend(selected_files_train)  # lista di stringhe
            test_files.extend(selected_files_test)    # lista di stringhe
    return train_files,test_files, report_train, report_test


def n_mosquito(data_path, files):
    n_mosq_arr = []
    for file in files:
        with np.load(Path(data_path) / Path(file), allow_pickle=True) as loaded__file:
            centroids = loaded__file["arr_1"]
        n_mosq = len(centroids)
        n_mosq_arr.append(n_mosq)
    return n_mosq_arr


def get_data_from_file_list(data_path, files):

    train_images = []
    train_bbox = []
    train_labels = []

    for file in files:
        with np.load(Path(data_path) / Path(file), allow_pickle=True) as loaded__file:

            images_train = loaded__file["arr_0"]
            bbox_train = loaded__file["arr_1"]
            label_train = loaded__file["arr_2"]

            train_images.extend(images_train)
            train_bbox.extend(bbox_train)
            train_labels.extend(label_train)
    return  np.array(train_images), train_bbox, train_labels


# +
def get_strig2num(test_labels):
    label_list = [d['species'] for d in test_labels]
    strig2num = {}
    string_classes = np.unique(label_list)
    numeric_classes = np.arange(len(string_classes))
    for key,value in zip(string_classes, numeric_classes, strict=True):
        strig2num[key] = value
    return strig2num

def get_target(test_labels,strig2num):

    y_test = np.array([strig2num[d['species']] for d in test_labels])
    #y_test = [strig2num[s] for s in label_list]
    return np.array(y_test)

def num2string(strig2num, test_values):
    reverse_dict = {value: key for key, value in strig2num.items()}
    return [reverse_dict.get(value) for value in test_values]



class BalancedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,num_workers = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Count the number of samples in each class
        class_counts = torch.bincount(torch.tensor(dataset.targets))

        # Compute the weight of each sample
        weights = 1.0 / class_counts[dataset.targets]# has the same length on the data array

        # Create a sampler that samples each class with equal probability
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        super().__init__(dataset, batch_size=batch_size, sampler=sampler,num_workers =num_workers )


class Mosaico_dataset(Dataset):
    def __init__(self, images,targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, target



