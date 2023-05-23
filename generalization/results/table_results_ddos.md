
| classifier | train dataset         | test dataset          | features                            | accuracy | std accuracy | precision | std precision | recall | std recall |
|------------|-----------------------|-----------------------|-------------------------------------|----------|--------------|-----------|---------------|--------|------------|
| rf         | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | all features                        | 0.997    | 0.0          | 0.997     | 0.0           | 0.998  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.495    | 0.002        | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.487    | 0.002        | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.499    | 0.0          | 0.051     | 0.152         | 0.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.5      | 0.0          | 0.007     | 0.02          | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.499    | 0.001        | 0.045     | 0.134         | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.474    | 0.004        | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | contaminants dropped                | 0.997    | 0.0          | 0.997     | 0.0           | 0.998  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.495    | 0.001        | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.517    | 0.105        | 0.095     | 0.284         | 0.07   | 0.211      |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.499    | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.499    | 0.001        | 0.099     | 0.137         | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.474    | 0.002        | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | generalization contaminants dropped | 0.883    | 0.001        | 0.882     | 0.004         | 0.885  | 0.004      |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.301    | 0.0          | 0.001     | 0.0           | 0.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.888    | 0.009        | 0.828     | 0.013         | 0.979  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.622    | 0.007        | 0.806     | 0.005         | 0.322  | 0.014      |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.975    | 0.0          | 0.969     | 0.001         | 0.982  | 0.001      |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.556    | 0.011        | 0.986     | 0.001         | 0.114  | 0.023      |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.489    | 0.0          | 0.493     | 0.0           | 0.832  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.58     | 0.0          | 0.543     | 0.0           | 0.998  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.802    | 0.0          | 0.716     | 0.0           | 1.0    | 0.0        |
|logistic    | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | all features                        | 0.921    | 0.0          | 0.905     | 0.0           | 0.941  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.417    | 0.0          | 0.415     | 0.0           | 0.409  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.724    | 0.0          | 0.683     | 0.0           | 0.836  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.374    | 0.0          | 0.399     | 0.0           | 0.497  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.999    | 0.0          | 0.999     | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.785    | 0.0          | 0.705     | 0.0           | 0.981  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.533    | 0.0          | 0.677     | 0.0           | 0.126  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.47     | 0.0          | 0.003     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.997    | 0.0          | 0.998     | 0.0           | 0.995  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | contaminants dropped                | 0.911    | 0.0          | 0.887     | 0.0           | 0.944  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.755    | 0.0          | 0.672     | 0.0           | 0.997  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.374    | 0.0          | 0.201     | 0.0           | 0.084  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.358    | 0.0          | 0.392     | 0.0           | 0.517  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.999    | 0.0          | 0.999     | 0.0           | 0.999  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.207    | 0.0          | 0.034     | 0.0           | 0.021  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.514    | 0.0          | 0.697     | 0.0           | 0.05   | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.472    | 0.0          | 0.001     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.996    | 0.0          | 0.998     | 0.0           | 0.995  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | generalizaion contaminants dropped  | 0.57     | 0.0          | 0.984     | 0.0           | 0.142  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.579    | 0.0          | 0.543     | 0.0           | 1.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.764    | 0.0          | 0.679     | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.645    | 0.0          | 0.822     | 0.0           | 0.371  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.89     | 0.0          | 0.896     | 0.0           | 0.882  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.763    | 0.0          | 0.679     | 0.0           | 0.998  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.802    | 0.0          | 0.716     | 0.0           | 0.998  | 0.0        |
|tabular_lr  | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | all features                        | 0.988    | 0.0          | 0.992     | 0.0           | 0.983  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.389    | 0.0          | 0.01      | 0.0           | 0.002  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.365    | 0.0          | 0.098     | 0.0           | 0.033  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.447    | 0.0          | 0.451     | 0.0           | 0.481  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.999    | 0.0          | 1.0       | 0.0           | 0.999  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.161    | 0.0          | 0.026     | 0.0           | 0.019  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.498    | 0.0          | 0.103     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.45     | 0.0          | 0.001     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | contaminants dropped                | 0.909    | 0.0          | 0.855     | 0.0           | 0.984  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.5      | 0.0          | 0.085     | 0.0           | 0.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.396    | 0.0          | 0.012     | 0.0           | 0.003  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.499    | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.998    | 0.0          | 0.996     | 0.0           | 0.999  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.25     | 0.0          | 0.041     | 0.0           | 0.022  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.5      | 0.0          | 0.232     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.499    | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.985    | 0.0          | 1.0       | 0.0           | 0.97   | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | generalization contaminants dropped | 0.559    | 0.0          | 0.532     | 0.0           | 0.977  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.447    | 0.0          | 0.001     | 0.0           | 0.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.735    | 0.0          | 0.653     | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.51     | 0.0          | 0.505     | 0.0           | 0.998  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.968    | 0.0          | 0.948     | 0.0           | 0.99   | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.603    | 0.0          | 0.558     | 0.0           | 0.997  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.498    | 0.0          | 0.049     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.501    | 0.0          | 0.736     | 0.0           | 0.002  | 0.0        |
|knn         | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | all features                        | 0.996    | 0.0          | 0.995     | 0.0           | 0.997  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.423    | 0.0          | 0.006     | 0.0           | 0.001  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.483    | 0.0          | 0.421     | 0.0           | 0.09   | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.552    | 0.0          | 0.735     | 0.0           | 0.164  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.407    | 0.0          | 0.061     | 0.0           | 0.013  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.766    | 0.0          | 0.894     | 0.0           | 0.602  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.295    | 0.0          | 0.001     | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
| knn        | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | contaminants dropped                | 0.9950   | 0.0          | 0.9940    | 0.0           | 0.9960 | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.4270   | 0.0          | 0.0110    | 0.0           | 0.0020 | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.6230   | 0.0          | 0.7460    | 0.0           | 0.3740 | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.6090   | 0.0          | 0.8450    | 0.0           | 0.2670 | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.4640   | 0.0          | 0.1300    | 0.0           | 0.0130 | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.7660   | 0.0          | 0.8970    | 0.0           | 0.6000 | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.2950   | 0.0          | 0.0020    | 0.0           | 0.0010 | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 1.0      | 0.0          | 1.0       | 0.0           | 1.0    | 0.0        |
|            | NF-ToN-IoT-V2         | NF-ToN-IoT-V2         | generalization contaminants dropped | 0.879    | 0.0          | 0.877     | 0.0           | 0.881  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.448    | 0.0          | 0.015     | 0.0           | 0.002  | 0.0        |
|            | NF-ToN-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.461    | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-BoT-IoT-V2         | NF-ToN-IoT-V2         |                                     | 0.47     | 0.0          | 0.481     | 0.0           | 0.765  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-BoT-IoT-V2         |                                     | 0.975    | 0.0          | 0.971     | 0.0           | 0.979  | 0.0        |
|            | NF-BoT-IoT-V2         | NF-CSE-CIC-IDS2018-V2 |                                     | 0.744    | 0.0          | 0.662     | 0.0           | 0.997  | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-ToN-IoT-V2         |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-BoT-IoT-V2         |                                     | 0.5      | 0.0          | 0.0       | 0.0           | 0.0    | 0.0        |
|            | NF-CSE-CIC-IDS2018-V2 | NF-CSE-CIC-IDS2018-V2 |                                     | 0.803    | 0.0          | 0.717     | 0.0           | 1.0    | 0.0        |
