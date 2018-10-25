# Reliablity Fusing Map (RFM)

This repository is implementation of "Image Tampering Detection and Localization via CNN-Based Reliablity Fusing Map (RFM)".

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

1. Download Drensden dataset into /code/dataset

2. For training process, running command:

   > python main.py --conf code.config --action train

   where `code.config.py` is config file including CNN architecture, dataset name, and so on.

3. For testing process, running command:

   > python main.py --conf code.config --action test

4. Also, you can make your personal tampering dataset:

   > python main.py --conf code.config --action splicing

<br>

# Result



![Results of comparative experiment with method proposed by Bondi et al. and our RFM method. (a)-(c) illustrates cover image, forged image, and ground truth image. (d) and (e) illustrates first iteration and output of RFM algorithm, (f) show result of Bondi et al.](https://github.com/grasses/Tampering-Detection-and-Localization/blob/master/static/result-2.jpg?raw=true)



Results of comparative experiment with method proposed by Bondi et al. and our RFM method. (a)-(c) illustrates cover image, forged image, and ground truth image. (d) and (e) illustrates first iteration and output of RFM algorithm, (f) show result of Bondi et al.

<br>

# License

This library is under the GPL V3 license. For the full copyright and license information, please view the LICENSE file that was distributed with this source code.