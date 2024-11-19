# Pytdoa: Multilateration for low cost SDR receivers in python

`pytdoa` is a library that allows for multilateration on low-cost, GPS-free receivers.

What `pytdoa` expects is a set of of raw iq signals belonging to a set of receivers with the target signal. Because we are not relying on GPS timing, the expected format of each of the raw iq files is:

```
-------------------------------------------------------------------------------
| Ref Signal (N samples) | Target Signal (N samples) | Ref Signal (N samples) |
-------------------------------------------------------------------------------

- Total samples: 3*N
```

We need the `Ref Signal` to be able to align samples. The last chunk can be also used to estimate the drift of the receiver over the whole sampling process. It is also **critical** that when switching frequencies, no samples were lost, otherwise this methodology cannot be applied.

## Citing
If you use this software please consider citing us:

```bibtex
@inproceedings{lizarribar2024oran,
  title={ORAN-Sense: Localizing Non-cooperative Transmitters with Spectrum Sensing and 5G O-RAN},
  author={Lizarribar, Yago and Calvo-Palomino, Roberto and Scalingi, Alessio and Santaromita, Giuseppe and Bovet, G{\'e}r{\^o}me and Giustiniano, Domenico},
  booktitle={IEEE INFOCOM 2024-IEEE Conference on Computer Communications},
  pages={1870--1879},
  year={2024},
  organization={IEEE}
}
```
