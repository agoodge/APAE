# APAE
## Adam Goodge, Bryan Hooi, See-Kiong Ng, Wee Siong Ng

# LUNAR
### Official Implementation of ["Robustness of Autoencoders for Anomaly Detection Under Adversarial Impact"](https://www.ijcai.org/Proceedings/2020/0173.pdf), Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (IJCAI2020) 

Detecting anomalies is an important task in a wide variety of applications and domains. Deep learning methods have achieved state-of-the-art performance in anomaly detection in recent years; unsupervised methods being particularly popular. However, deep learning methods can be fragile to small perturbations in the input data. This can be exploited by an adversary to deliberately hinder model performance; an adversarial attack. This phenomena has been widely studied in the context of supervised image classification since its discovery, however such studies for an anomaly detection setting are sorely lacking. Moreover, the plethora of defense mechanisms that have been proposed are often not applicable to unsupervised anomaly detection models. In this work, we study the effect of adversarial attacks on the performance of anomalydetecting autoencoders using real data from a Cyber physical system (CPS) testbed with intervals of controlled, physical attacks as anomalies. An adversary would attempt to disguise these points as normal through adversarial perturbations. To combat this, we propose the Approximate Projection Autoencoder (APAE), which incorporates two defenses against such attacks into a general autoencoder. One of these involves a novel technique to improve robustness under adversarial impact by optimising latent representations for better reconstruction outputs.

## Files
main.py \
attack.py - adversarial attacks\
defend.py - defenses against attacks\
model.py - Autoencoder model and training \
variables.py - hyperparameters \
utils.py - useful functions


## Citation
```
@inproceedings{goodge2020robustness,
  title={Robustness of Autoencoders for Anomaly Detection Under Adversarial Impact.},
  author={Goodge, Adam and Hooi, Bryan and Ng, See-Kiong and Ng, Wee Siong}
  journal={29th International Joint Conference on Artificial Intelligence, IJCAI 2020}
  year={2020}
}
