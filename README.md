# Reliability Fusing Map (RFM)

This repository is implementation of "Image Tampering Detection and Localization via Reliability Fusion Map” (RFM). The main contributions are summarized as follows  (1) obtaining higher accuracy; (2) reducing computational complexity of clustering; (3) improving localization fineness from 64 x 64 to 32 x 32. 

<br>

# Prerequisites

* tensorflow == 1.7.0
* pandas == 0.23.4
* scipy == 1.1.0
* sklearn == 0.19.2
* matplotlib == 2.2.3
* Pillow == 5.2.0

<br>

# Usage

* Run pretrain model:

1. You can download pretrain model at: [Baidu disk](https://pan.baidu.com/s/1mYEHwtQdIUb5vugruUppRA), password=`mxqr`
   [Google drive](https://drive.google.com/file/d/1ULTmA1Ef5Y8NcOc1bSF8Ksj7rapaN0zO/view?usp=sharing)

   put unzip folder into `code/model/{scope_name}`, see `{scope_name}` in `code.config.py`

2. Run pre-train test using command:

   > python main.py --code.config --action test

   where `code.config.py` is config file including CNN architecture, dataset name, and so on. <br>

   

   The CNN module pre-train output is a csv file, which format with: 

   `{f1,f2,f3...,predict_label,true_label,quality_factory}`, where f1,f2... is CNN confidence of each camera model.

3. Post-train in `experiment` folder

<hr><br>

* You can train your personal pretrain model:

1. Download Drensden dataset into /code/dataset

   > python main.py --code.config --action download --name Dresden

2. Generate a tensorflow `records` file:

   > python main.py --code.config --action generator

3. For training process, running command:

   > python main.py --conf code.config --action train

4. Making your personal tampering dataset:

   > python main.py --conf code.config --action splicing

<br>

# Result



![Results of comparative experiment with method proposed by Bondi et al. and our RFM method. (a)-(c) illustrates cover image, forged image, and ground truth image. (d) and (e) illustrates first iteration and output of RFM algorithm, (f) show result of Bondi et al.](https://github.com/grasses/Tampering-Detection-and-Localization/blob/master/static/result-3.jpg?raw=true)



Results of comparative experiment with method proposed by Bondi et al. and our RFM method. (a)-(c) illustrates cover image, forged image, and ground truth image. (d) and (e) illustrates result of cover image and forged image with RFM algorithm, (f) show result of Bondi et al. We improve localization fineness from 64 x 64 to 32 x 32 (see (e) and (f)).

<br>

# License

This library is under the GPL V3 license. For the full copyright and license information, please view the LICENSE file that was distributed with this source code.
