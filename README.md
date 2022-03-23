[ecu]: https://documents.uow.edu.au/~phung/download.html "ECU download page"


# Skinny-Bench
Calculating inference time of the Skinny network.  

#### Original Paper
T. Tarasiewicz, J. Nalepa, and M. Kawulok. “Skinny: A Lightweight U-net for Skin Detection and Segmentation”. In: 2020 IEEE International Conference on Image Processing (ICIP). IEEE. 2020, pp. 2386–2390. https://doi.org/10.1109/ICIP40778.2020.9191209.

#### Credits
Credits to the authors of the original work: 
https://github.com/ttarasiewicz/Skinny


## Performance

Measured inference time: `0.242685 ± 0.016` seconds.

#### Improvements with respect to thesis
[slow]: https://github.com/tensorflow/tensorflow/issues/39458 "Keras predict is slow on first call"
The inference time recorded in the thesis is worse because the model
was re-loaded prior to each prediction, and [Keras predict is slow on first call][slow]
because the predict function is compiled during the first (and only the first) call to predict.  
By loading the model before performing the predictions, and dumping a first prediction
before starting the observations, the inference time is greatly improved, 
outperforming the probabilistic approach featured in the thesis.


## Methodology

#### Rules
The inference time evaluation follows these rules:
- Image loading into memory is excluded.
- Image saving to disk is excluded.
- The measurement starts when the algorithm starts.
- Pre-processing and post-processing, if present, are included in the measured execution time.

The first 14 images of the ECU dataset, which are all the same size (352×288), are used as
the performance evaluation set.  
One image at a time is processed by the skin detector and the resulting execution time
is measured.  
The evaluation set is processed 5 times and, each time, the average
measurement time is calculated.  
Finally, the five average values are averaged into a single value
and the standard deviation is computed.


#### System
Inference time measurements were all performed on an i7 4770k processor
running on Pop!_OS 20.10 x86_64 with 16 GB of RAM.  
The experiments were performed using Python 3.8.6 64bit along with the
packages listed in the requirements file, specifically with Tensorflow 2.5.0.
Whereas models were trained using Google Colab with Tensorflow 2.4.1 and Python 3.7.10.

## Usage

[Download (ask the authors)][ecu] the ECU [1] dataset and place it into the folder `dataset`.  
To work properly, the dataset must respect the following:
- origin images are placed in `dataset/ECU/origin_images`
- origin images are named: `im00001.jpg`, `im00002.jpg`, `im00003.jpg`, ..
- mask images are placed in `dataset/ECU/skin_masks`
- mask images are named: `im00001.png`, `im00002.png`, `im00003.png`, ..

Install the pip requirements and run the `main.py` file.

| Ref   | Publication |
| :---  | :--- |
| 1     | Phung, S., Bouzerdoum, A., & Chai, D. (2005). Skin segmentation using color pixel classification: analysis and comparison. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(1), 148-154. https://doi.org/10.1109/tpami.2005.17  |
