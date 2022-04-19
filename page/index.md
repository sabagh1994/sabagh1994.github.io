<!-- ==============================
     The Paper's Premise
     ============================== -->

\begin{section}{title="The Premise", name="Premise"}

Every time you train a logistic classifier in a few-shot task and minimize a cross-entropy loss, you're essentially performing a **Maximum Likelihood Estimation (MLE)**.

In the world of statistics, MLEs are known to suffer from serious bias issues when only few samples are provided.

The few shots you use are random samples, so the MLEs are going to be random variables too. There is no easy way around this stochasticity and you would always get some randomness or variance in the resulting MLE.

That being said, it's only reasonable to hope that you'd get the right parameters with MLE, at least on average. Well, that's the issue! Not only MLEs can have a lot of variance, but also they can be severely off even on-average!

\begin{center}
\figure{path="/assets/static_figures/mlebiasslide.svg", width="100%", style="border-radius:5px;", caption="A conceptual visualization of the classifier parameter estimation's bias problem in few-shot learning."}
\end{center}

**_Care for a simple example to see the severity of this issue?_**

Here is a simple toy-example show-casing this issue in a [few-shot geometric experiment with a fair coin](https://en.wikipedia.org/wiki/Geometric_distribution) (yes; the same exact problem from your introductory probability course).

You want to recover the coin head probability with a few number of samples, so you use the MLE. However, you're curious if you'd even get the right parameter on average. To check that, you simulate some experiments in python and plot the average MLE. Here's what you'll see:

\begin{center}
\begin{columns}
\begin{column}{}
_The Average MLE in a Geometric Experiment_

\html{<img src="/assets/static_figures/avgmle_vs_nsamples_geom.svg" alt="drawing" width="100%"/>}

The \style{color:blue;}{blue} points show the average MLE for various sample sizes.
The **black** double-sided vertical arrow shows the MLE bias away from the true parameter.
The \style{color:red;}{red} points show a slightly better estimator than MLE.
\end{column}
\begin{column}{}
_The MLE Bias vs. the Sample Size in a log-log Scale_

\html{<img src="/assets/static_figures/logmlebias_vs_lognsamples_geom.svg" alt="drawing" width="100%"/>}

The vertical axis shows the log-10 of the MLE bias, and the horizontal axis shows the log-10 of the number of samples. The bias isn't going away as fast as you would hope so; it's of $O(N^{-1})$ and far from an exponential drop for sure.
\end{column}

\end{columns}
\end{center}

This begs the central question in our paper:

\begin{center}
__If MLEs cannot recover the true parameter even on average in such an easy problem,
then how can we trust they're best for few-shot logistic classifiers with thousands of dimensions?__
\end{center}

\begin{center}\end{center}
This motivates the introduction of the Firth MLE penalty, a glimpse of which was shown in the geometric example plot in red.

\end{section}

<!-- ==============================
     Firth Bias Reduction in Few Words
     ============================== -->
\begin{section}{title="Firth Bias Reduction in Few Words", name="Theory"}
**For 1-Layer Logistic and Cosine Classifiers with the Cross-Entropy Loss**:

All you need to do, is replace
$$\hat{\beta} = \text{argmin}_{\beta} \quad \frac{1}{N}\sum_{i=1}^{N} \bigg[\text{CE}(\mathbf{P}_i, \mathbf{y}_i)\bigg]$$
with
$$\hat{\beta}_{\text{Firth}} = \text{argmin}_{\beta} \quad \frac{1}{N}\sum_{i=1}^{N} \bigg[\text{CE}(\mathbf{P}_i, \mathbf{y}_i) + \lambda \cdot \text{CE}(\mathbf{P}_i,\mathbf{U}) \bigg]$$
where $\mathbf{U}$ is the uniform distribution over the classes, and $\lambda$ is a positive constant. The CE-term with the uniform distribution is basically the (negative) sum of the prediction log-probability values over all data points and classes.

[Our paper](https://arxiv.org/abs/2110.02529) provides a theoretical proof of why the added penalty is a simplification of a $\log(\det(F))$ term (thereby, encouraging "larger" Fisher information).

**General Firth Bias Reduction Form**:

Add a log-det of FIM term to your loss minimization problem. That is, replace
$$\hat{\beta}=\text{argmin}_{\beta}\quad\bigg[l(\beta)\bigg]$$
with
$$\hat{\beta}_{\text{Firth}} = \text{argmin}_{\beta} \quad \bigg[l(\beta) + \lambda\cdot \log(\det(F))\bigg]$$
This was proven to reduce the bias of your estimated parameters in [Firth's original work](https://www.jstor.org/stable/pdf/2336755.pdf?casa_token=PlU8RYXqYMcAAAAA:Zawbrw_XhF36J9M9Ht1oD4AAScYxGIgh5APJq6XFWV_BhIOFxlYVBIY4pKipBvGaJNhFRXNcXbWB2JxmjYMwpxISq0os40RfvXA5Cbrso20VMu9XlrI)

\end{section}


<!-- ==============================
     Experiments
     ============================== -->
\begin{section}{title="Experiments and Results", name="Results"}
**Logistic Classifiers and Basic Feature Backbones**

The following is the effect of Firth bias reduction compared to typical L2 regularization in 16-way few-shot classification tasks using basic feature backbones and 1-layer logistic classifiers. The vertical axis shows the accuracy improvements, and the horizontal axis shows the number of shots.

\begin{center}
\begin{columns}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nshots_firth_1layer_mini.svg" alt="drawing" width="100%" background="{{white_bg_color}}"/>}
\end{column}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nshots_l2_1layer_mini.svg" alt="drawing" width="100%"/>}
\end{column}
\end{columns}
\end{center}

\begin{center}\end{center}
Here's the same set of results, but with 3-layer logistic classifiers (instead of 1-layer networks).

\begin{center}
\begin{columns}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nshots_firth_3layer_mini.svg" alt="drawing" width="100%"/>}
\end{column}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nshots_l2_3layer_mini.svg" alt="drawing" width="100%"/>}
\end{column}
\end{columns}
\end{center}


\begin{center}\end{center}
**Cosine Classifiers and S2M2R Feature Backbones**

Below is the effect of Firth bias reduction on cosine classifiers and S2M2R features. The horizontal axis is the number of classes, and the vertical axis shows the Firth accuracy improvements.

\begin{center}
\begin{columns}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nways_miniImagenet.svg" alt="drawing" width="100%"/>}
\end{column}
\begin{column}{}
\html{<img src="/assets/static_figures/dacc_vs_nways_cifar.svg" alt="drawing" width="100%"/>}
\end{column}
\end{columns}
\end{center}

\begin{center}
\begin{columns}
\begin{column}{}
\end{column}
\html{<img src="/assets/static_figures/dacc_vs_nways_tieredImagenet.svg" alt="drawing" width="100%"/>}
\end{columns}
\end{center}

\begin{center}\end{center}
**Firth Bias Reduction on the Distribution Calibration Method**

The following shows the recent state of the art method of few-shot [Distribution Calibration (DC)](https://arxiv.org/abs/2101.06395) in cross-domain settings with and without Firth bias reduction. Each setting was tested with and without data augmentation (addition of 750 samples), and the maximum accuracy was reported. Note that the confidence intervals are much smaller for the improvement column, thanks to the random-effect matching procedure we used in this study.
\begin{center}
\begin{table}{class="table-striped"}
|     	|        |             |     mini &#8594; CUB 	|              	 |             |   tiered &#8594; CUB   |           	    |
|:-----:|:------:|:-----------:|:---------------------:|:---------------:|:-----------:|:----------------------:|:---------------:|
|**Way**|**Shot**| **Before**  |        **After**      | **Improvement** |  **Before** |         **After**      | **Improvement** |
| 10  	| 1      | 37.14 ± 0.12| 37.41 ± 0.12          | 0.27 ± 0.03     | 64.36 ± 0.16| 64.52 ± 0.16          	| 0.15 ± 0.03     |
| 10  	| 5    	 | 59.77 ± 0.12| 60.77 ± 0.12          | 1.00 ± 0.04     | 86.23 ± 0.10| 86.66 ± 0.09          	| 0.43 ± 0.03     |
| 15  	| 1    	 | 30.22 ± 0.09| 30.37 ± 0.09          | 0.15 ± 0.03     | 57.73 ± 0.13| 57.73 ± 0.13          	| 0.00 ± 0.00     |
| 15  	| 5    	 | 52.73 ± 0.09| 53.84 ± 0.09          | 1.11 ± 0.03     | 82.16 ± 0.09| 83.05 ± 0.08          	| 0.89 ± 0.03     |
\end{table}
\end{center}


\end{section}

<!-- ==============================
     GETTING STARTED
     ============================== -->
\begin{section}{title="Implementation", name="Implementation"}

Implementing Firth bias reduction for 1-layer logistic and cosine classifiers only takes one or two extra lines of code.
```python
ce_loss = nn.CrossEntropyLoss()
ce_term = ce_loss(logits, target)

# This is how you can compute the Firth bias reduction from classifier logits
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
firth_term = -log_probs.mean()

loss = ce_term + lam * firth_term
loss.backward()
```
Alternatively, you can use the `label_smoothing` keyword argument in [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Remember that this Firth formulation is only true for 1-layer logistic and cosine classifiers. For more complex networks, the FIM's log-determinant must be worked out.

As for the $\lambda$ coefficient,
 * Firth's original work set it to a pre-determined constant.
 * Recently, [$\log F(m,m)$ models](https://onlinelibrary.wiley.com/doi/10.1002/sim.6537) proposed scaling Firth's pre-determined coefficient, making $\lambda$ a hyper-parameter.
 * We followed the common machine learning practice, and validated the $\lambda$ coefficient on the validation split, then evaluated the validated $\lambda$ on the novel set.
 * You don't need much resolution for the validation search; we performed the $\lambda$ search in a log-10 space on a handful of candidates ($\lambda\in \{0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0\}$).

\end{section}

<!-- ==============================
     Code
     ============================== -->
\begin{section}{title="Code", name="Code"}

Our implementation is open-source and available at [https://github.com/ehsansaleh/firth\_bias\_reduction](https://github.com/ehsansaleh/firth_bias_reduction).

Due to the volume of experimental settings in our paper, we broke down the code into three sub-modules:

* The [`code_firth`](https://github.com/ehsansaleh/code_firth) repository corresponds to the Firth bias reduction experiments using standard ResNet architectures and logistic classifiers (e.g., Figure 2 and 3 in the main paper).

* The [`code_s2m2rf`](https://github.com/ehsansaleh/code_s2m2rf) repository corresponds to the experiments with cosine classifiers on WideResNet-28 feature stacks trained by the S2M2R method.

* The [`code_dcf`](https://github.com/sabagh1994/code_dcf) repository contains our GPU implementation of the [Distribution Calibration (DC)](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration) method and the relevant Firth bias reduction improvements.

All of them are standalone repositories with
* detailed documentation in their corresponding readme files, and
* helper scripts for automatically downloading and extracting the features,
  datasets, and backbone parameters from the external sources (such as Google Drive).

You can clone all three modules with the following command:

```bash
git clone --recursive https://github.com/ehsansaleh/firth_bias_reduction.git
```

\end{section}

<!-- ==============================
     Data
     ============================== -->
\begin{section}{title="Data", name="Data"}

We have published all the data, pre-computed features, trained backbone parameters,
and other auxiliary files (complimenting the open-source code) in two redundant
external sources; the Illinois Data Bank and Google-Drive.

\alert{Our [open-source code repositories](https://github.com/ehsansaleh/firth_bias_reduction) include **automated downloading shell scripts** in each module that are programmed to pull the data from the external sources and extract and verify their integrity. Ideally, those scripts should automatically download and populate the workspace for you to reproduce our results without any manual intervention. The following details are only referenced for fail-safe redundancy and citation purposes.}

**Illinois Data Bank**

We have included our pre-computed features and and trained backbones in tar-ball archives in our Illinois Data Bank Repository at [{{idb_url}}]({{idb_url}}) with brief instructions for manually downloading and placing the data.

**Google Drive**

<!-- * The [code_firth](https://github.com/ehsansaleh/code_firth) module
  * For populating [the features directory](https://github.com/ehsansaleh/code_firth/tree/main/features) use the following URLs:
    * [https://drive.google.com/drive/folders/1mKUg4ifex8BDU76bnL-4b4gMwoFKHBC4?usp=sharing](https://drive.google.com/drive/folders/1mKUg4ifex8BDU76bnL-4b4gMwoFKHBC4?usp=sharing)
    * [https://drive.google.com/drive/folders/11pbf2kSJ6XYjKxmY-plITicS99_yvD_g?usp=sharing](https://drive.google.com/drive/folders/11pbf2kSJ6XYjKxmY-plITicS99_yvD_g?usp=sharing)
  * For populating [the backbones directory](https://github.com/ehsansaleh/code_firth/tree/main/backbones) use the following URL:
    * [https://drive.google.com/drive/folders/1nw7YKg7O0BEI9Qm3KeSyKqSxh0BslrWe?usp=sharing](https://drive.google.com/drive/folders/1nw7YKg7O0BEI9Qm3KeSyKqSxh0BslrWe?usp=sharing)
  * For populating [the datasets directory](https://github.com/ehsansaleh/code_firth/tree/main/datasets) use the following URLs:
    * [https://drive.google.com/drive/folders/1IyMLo10ngferRtRC6GWaw3wwjTInWbia?usp=sharing](https://drive.google.com/drive/folders/1IyMLo10ngferRtRC6GWaw3wwjTInWbia?usp=sharing)
    * [https://drive.google.com/drive/folders/1wC175y0pGitRjNbbrABw_cz5IxNP4B01?usp=sharing](https://drive.google.com/drive/folders/1wC175y0pGitRjNbbrABw_cz5IxNP4B01?usp=sharing)


* The [code_s2m2rf](https://github.com/ehsansaleh/code_s2m2rf) module
  * For populating [the features directory](https://github.com/ehsansaleh/code_s2m2rf/tree/main/features) use the following URLs:
    * [https://drive.google.com/file/d/1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf/view?usp=sharing](https://drive.google.com/file/d/1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf/view?usp=sharing)
  * For populating [the filelists directory](https://github.com/ehsansaleh/code_s2m2rf/tree/main/filelists) use the following URLs:
    * [https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)
  * For populating [the checkpoints directory](https://github.com/ehsansaleh/code_s2m2rf/tree/main/checkpoints) use the following URL:
    * [https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing](https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing)
  * For populating [the Datasets directory](https://github.com/ehsansaleh/code_s2m2rf/tree/main/Datasets) use the following URL:
    * [https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing](https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing)


* The [code_dcf](https://github.com/sabagh1994/code_dcf) module
  * For populating [the features directory](https://github.com/sabagh1994/code_dcf/tree/main/features) use the following URL:
    * [https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing](https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing)
  * For populating [the cache directory](https://github.com/sabagh1994/code_dcf/tree/main/cache) use the following URL:
    * [https://drive.google.com/file/d/1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU/view?usp=sharing](https://drive.google.com/file/d/1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU/view?usp=sharing)
  * For populating [the filelists directory](https://github.com/sabagh1994/code_dcf/tree/main/filelists) use the following URL:
    * [https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)
  * For populating [the checkpoints directory](https://github.com/sabagh1994/code_dcf/tree/main/checkpoints) use the following URL:
    * [https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing](https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing)
  * For populating [the Datasets directory](https://github.com/sabagh1994/code_dcf/tree/main/Datasets) use the following URL:
    * [https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing](https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing) -->

<!-- \begin{table}{class="table-striped"}
|    Module   |    Directory   |      URLs   |
|:------------|:---------------|:------------|
|[code_firth](https://github.com/ehsansaleh/code_firth)|[features](https://github.com/ehsansaleh/code_firth/tree/main/features)|[URL1](https://drive.google.com/drive/folders/1mKUg4ifex8BDU76bnL-4b4gMwoFKHBC4?usp=sharing) and [URL2](https://drive.google.com/drive/folders/11pbf2kSJ6XYjKxmY-plITicS99_yvD_g?usp=sharing)|
|[code_firth](https://github.com/ehsansaleh/code_firth)|[backbones](https://github.com/ehsansaleh/code_firth/tree/main/backbones)|[URL](https://drive.google.com/drive/folders/1nw7YKg7O0BEI9Qm3KeSyKqSxh0BslrWe?usp=sharing)|
|[code_firth](https://github.com/ehsansaleh/code_firth)|[datasets](https://github.com/ehsansaleh/code_firth/tree/main/datasets)|[URL1](https://drive.google.com/drive/folders/1IyMLo10ngferRtRC6GWaw3wwjTInWbia?usp=sharing) and [URL2](https://drive.google.com/drive/folders/1wC175y0pGitRjNbbrABw_cz5IxNP4B01?usp=sharing)|
|[code_s2m2rf](https://github.com/ehsansaleh/code_s2m2rf)|[features](https://github.com/ehsansaleh/code_s2m2rf/tree/main/features)|[URL](https://drive.google.com/file/d/1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf/view?usp=sharing)|
|[code_s2m2rf](https://github.com/ehsansaleh/code_s2m2rf)|[filelists](https://github.com/ehsansaleh/code_s2m2rf/tree/main/filelists)|[URL](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)|
|[code_s2m2rf](https://github.com/ehsansaleh/code_s2m2rf)|[checkpoints](https://github.com/ehsansaleh/code_s2m2rf/tree/main/checkpoints)|[URL](https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing)|
|[code_s2m2rf](https://github.com/ehsansaleh/code_s2m2rf)|[Datasets](https://github.com/ehsansaleh/code_s2m2rf/tree/main/Datasets)|[URL](https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing)|
|[code_dcf](https://github.com/sabagh1994/code_dcf)|[features](https://github.com/sabagh1994/code_dcf/tree/main/features)|[URL](https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing)|
|[code_dcf](https://github.com/sabagh1994/code_dcf)|[cache](https://github.com/sabagh1994/code_dcf/tree/main/cache)|[URL](https://drive.google.com/file/d/1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU/view?usp=sharing)|
|[code_dcf](https://github.com/sabagh1994/code_dcf)|[filelists](https://github.com/sabagh1994/code_dcf/tree/main/filelists)|[URL](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)|
|[code_dcf](https://github.com/sabagh1994/code_dcf)|[checkpoints](https://github.com/sabagh1994/code_dcf/tree/main/checkpoints)|[URL](https://github.com/sabagh1994/code_dcf/tree/main/checkpoints)|
|[code_dcf](https://github.com/sabagh1994/code_dcf)|[Datasets](https://github.com/sabagh1994/code_dcf/tree/main/Datasets)|[URL](https://github.com/sabagh1994/code_dcf/tree/main/Datasets)|
\end{table} -->


\begin{columns}
\begin{column}{}
  \begin{center}The [`code_firth`](https://github.com/ehsansaleh/code_firth) module\end{center}
  \begin{table}{class="table-striped"}
  |    Directory   |      Links  |
  |:---------------|:------------|
  |[`features`](https://github.com/ehsansaleh/code_firth/tree/main/features)|[URL1](https://drive.google.com/drive/folders/1mKUg4ifex8BDU76bnL-4b4gMwoFKHBC4?usp=sharing) and [URL2](https://drive.google.com/drive/folders/11pbf2kSJ6XYjKxmY-plITicS99_yvD_g?usp=sharing)|
  |[`backbones`](https://github.com/ehsansaleh/code_firth/tree/main/backbones)|[URL](https://drive.google.com/drive/folders/1nw7YKg7O0BEI9Qm3KeSyKqSxh0BslrWe?usp=sharing)|
  |[`datasets`](https://github.com/ehsansaleh/code_firth/tree/main/datasets)|[URL1](https://drive.google.com/drive/folders/1IyMLo10ngferRtRC6GWaw3wwjTInWbia?usp=sharing) and [URL2](https://drive.google.com/drive/folders/1wC175y0pGitRjNbbrABw_cz5IxNP4B01?usp=sharing)|
  \end{table}
\end{column}
\begin{column}{}
  \begin{center}The [`code_s2m2rf`](https://github.com/ehsansaleh/code_s2m2rf) module\end{center}
  \begin{table}{class="table-striped"}
  |    Directory   |      Links  |
  |:---------------|:------------|
  |[`features`](https://github.com/ehsansaleh/code_s2m2rf/tree/main/features)|[URL](https://drive.google.com/file/d/1Z-oO1hvcZkwsCZo9n7R3UGwHvFLi6Arf/view?usp=sharing)|
  |[`filelists`](https://github.com/ehsansaleh/code_s2m2rf/tree/main/filelists)|[URL](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)|
  |[`checkpoints`](https://github.com/ehsansaleh/code_s2m2rf/tree/main/checkpoints)|[URL](https://drive.google.com/drive/folders/1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_?usp=sharing)|
  |[`Datasets`](https://github.com/ehsansaleh/code_s2m2rf/tree/main/Datasets)|[URL](https://drive.google.com/drive/folders/1hJZ8akZ7krtka4x31YVOCkFn9s3VHq-I?usp=sharing)
  \end{table}
\end{column}
\begin{column}{}
  \begin{center}The [`code_dcf`](https://github.com/sabagh1994/code_dcf) module\end{center}
  \begin{table}{class="table-striped"}
  |    Directory   |     Links   |
  |:---------------|:------------|
  |[`features`](https://github.com/sabagh1994/code_dcf/tree/main/features)|[URL](https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing)|
  |[`cache`](https://github.com/sabagh1994/code_dcf/tree/main/cache)|[URL](https://drive.google.com/file/d/1w9_YOLgIVhtN8YwdiyAkdLLeIVB5UVuU/view?usp=sharing)|
  |[`filelists`](https://github.com/sabagh1994/code_dcf/tree/main/filelists)|[URL](https://drive.google.com/file/d/1AnakHzG3tf-ijT8udNDaqx_nbJRKflw6/view?usp=sharing)|
  |[`checkpoints`](https://github.com/sabagh1994/code_dcf/tree/main/checkpoints)|[URL](https://github.com/sabagh1994/code_dcf/tree/main/checkpoints)|
  |[`Datasets`](https://github.com/sabagh1994/code_dcf/tree/main/Datasets)|[URL](https://github.com/sabagh1994/code_dcf/tree/main/Datasets)|
  \end{table}
\end{column}
\end{columns}


\end{section}

<!-- ==============================
     References
     ============================== -->

\begin{section}{title="References", name="References"}
* Here is the arxiv link to our paper:
  * The arxiv PDF link: [https://arxiv.org/pdf/2110.02529.pdf](https://arxiv.org/pdf/2110.02529.pdf)
  * The arxiv web-page link: [https://arxiv.org/abs/2110.02529](https://arxiv.org/abs/2110.02529)
* Here is the open-review link to our paper:
  * The open-review PDF link: [https://openreview.net/pdf?id=DNRADop4ksB](https://openreview.net/pdf?id=DNRADop4ksB)
  * The open-review forum link: [https://openreview.net/forum?id=DNRADop4ksB](https://openreview.net/forum?id=DNRADop4ksB)
* Our paper got a spotlight presentation at ICLR 2022.
  * We will update here with links to the presentation video and the web-page on `iclr.cc`.

Here is the bibtex citation entry for our work:
```
@inproceedings{ghaffari2022fslfirth,
    title={On the Importance of Firth Bias Reduction In Few-Shot Classification},
    author={Saba Ghaffari and Ehsan Saleh and David Forsyth and Yu-Xiong Wang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=DNRADop4ksB}
}
```
\end{section}
