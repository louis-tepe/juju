# APTOS 2019 Blindness Detection Dataset

## Description

You are provided with a large set of retina images taken using fundus photography under a variety of imaging conditions.

A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

- **0** - No DR
- **1** - Mild
- **2** - Moderate
- **3** - Severe
- **4** - Proliferative DR

> **Note:** Like any real-world data set, you will encounter noise in both the images and labels. Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

## Files

In a synchronous Kernels-only competition, the files you can observe and download will be different than the private test set and sample submission.

| File                    | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `train.csv`             | The training labels                                                     |
| `test.csv`              | The test set (you must predict the diagnosis value for these variables) |
| `sample_submission.csv` | A sample submission file in the correct format                          |
| `train.zip`             | The training set images                                                 |
| `test.zip`              | The public test set images                                              |

### Public vs Private

The files may have different ids, may be a different size, and may vary in other ways, depending on the problem. You should structure your code so that it returns predictions for the public test set images in the format specified by the public `sample_submission.csv`, but does not hard code aspects like the id or number of rows.

When Kaggle runs your Kernel privately, it substitutes the private test set and sample submission in place of the public ones. You can plan on the private test set consisting of **20GB of data across 13,000 images** (approximately).

## Metadata

| Property        | Value                        |
| --------------- | ---------------------------- |
| **Total Files** | 5593 files                   |
| **Size**        | 10.22 GB                     |
| **Type**        | png, csv                     |
| **License**     | Subject to Competition Rules |

### Statistics (Train)

| Diagnosis         | Count |
| ----------------- | ----- |
| 0 (No DR)         | 1,928 |
| 1 (Mild)          | 370   |
| 2 (Moderate)      | 999   |
| 3 (Severe)        | 193   |
| 4 (Proliferative) | 295   |
