[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC.git/master)
<h1> Repo under review and construction </h1>

Python version: 3.8.4</p>

<h2>Experiment list</h3>
<p> The experiments are located under the "experiments/notebooks" folder. Please launch the binder link above or download the repository and run it locally.
Requirements are listed in the "requirements.txt" file. If run via binder the limitations of that environment (2 cores + 2 GB ram).</p>
<p> It is advised to run only the visualisation notebooks for each experiment. For those, please look for the "-VIS" suffix in the notebook's filename.</p>
</p>
<p>NotebookMatcher usedDatasetExperiment discussionContentsDirect binder link</p>
<table width="590">
<tbody>
<tr>
<td style="font-weight: 400;">ALG_000_EXP_001_Metrics_validation.ipynb</td>
<td style="font-weight: 400;">-</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Unit tests run, then local implementation of Middlebury metrics implementation is cross-checked with published results.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_000_EXP_001_Metrics_validation.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_001-EXP_001-Baseline-VIS.ipynb</td>
<td style="font-weight: 400;">OriginalMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">3 diffferent initialisation methods are tested, produced disparity maps are output at the end of the experiment.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_001-EXP_001-Baseline-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_002_EXP_001-Baseline-MacLean_et_al.ipynb</td>
<td style="font-weight: 400;">OriginalMatcher3</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Implementing disparity constraints for runtime improvements proposed by MacLean, W. J., Sabihuddin, S. and Islam, J. (2010).</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_002_EXP_001-Baseline-MacLean_et_al.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_002_EXP_001-VIS.ipynb</td>
<td style="font-weight: 400;">OriginalMatcher3</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Runtime and error comparision with baseline, histograms, "hit n miss" errors, 3d visualisation of disparities with intensity values back projected as their surface.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_002_EXP_001-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_003_EXP_001-Baseline-MacLean_et_al-Parallel_version.ipynb</td>
<td style="font-weight: 400;">ParallelMatcherNWMacLean</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Scanline-wise parallelisation of the pipeline in python. Disparity outputs displayed.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_003_EXP_001-Baseline-MacLean_et_al-Parallel_version.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_003_EXP_001-VIS.ipynb</td>
<td style="font-weight: 400;">ParallelMatcherNWMacLean</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Runtime and error comparision with baseline,&nbsp; output log file.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_003_EXP_001-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_004_EXP_001-Baseline-MacLean_et_al-Numba.ipynb</td>
<td style="font-weight: 400;">NumbaSimpleMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Converting the pipeline to Numba. Disparity outputs displayed.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_004_EXP_001-Baseline-MacLean_et_al-Numba.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_004_EXP_001-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaSimpleMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Runtime and error comparision with baseline, parallel baseline, output log file.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_004_EXP_001-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_004_EXP_002-Baseline-MacLean_et_al-param_search.ipynb</td>
<td style="font-weight: 400;">NumbaSimpleMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Running the pipeline with a range of match values (1-60).</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_004_EXP_002-Baseline-MacLean_et_al-param_search.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_004_EXP_002-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaSimpleMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.Tabular output.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_004_EXP_002-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_005_EXP_001-PatchMatch-2003.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Implementing constant-weight cost aggregation into the pipeline. A set of window sizes are tested. Tabular output of logged results.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_005_EXP_001-PatchMatch-2003.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_005_EXP_001-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_005_EXP_001-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_005_EXP_002-PatchMatch-2003.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Testing a set of support windows, used edge detectors and gaussian weights. Weights dipslayed, results are in tabular format.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_005_EXP_002-PatchMatch-2003.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_005_EXP_002-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_005_EXP_002-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_005_EXP_003-PatchMatch_2014-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2014</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_005_EXP_003-PatchMatch_2014-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_001-Bilateral_product_2003.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcher</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_001-Bilateral_product_2003.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_002-Bilateral_summed_2003.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcherBilateral</td>
<td style="font-weight: 400;">Middlebury 2003</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_002-Bilateral_summed_2003.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_003-Bilateral_product_2014-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcherBilateral</td>
<td style="font-weight: 400;">Middlebury 2014</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_003-Bilateral_product_2014-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_004-Bilateral_summed_2014-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcherBilateral</td>
<td style="font-weight: 400;">Middlebury 2014</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_004-Bilateral_summed_2014-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_005-Comparative_analysis-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcherBilateral</td>
<td style="font-weight: 400;">Middlebury 2014</td>
<td style="font-weight: 400;">Yes</td>
<td style="font-weight: 400;">Comparative interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_005-Comparative_analysis-VIS.ipynb)</td>
</tr>
<tr>
<td style="font-weight: 400;">ALG_006_EXP_006-Bilateral_sum_truncated_2014-VIS.ipynb</td>
<td style="font-weight: 400;">NumbaPatchMatcherBilateral (with truncated cost)</td>
<td style="font-weight: 400;">Middlebury 2014</td>
<td style="font-weight: 400;">No</td>
<td style="font-weight: 400;">Interactive visual experiment analysis using plotly.</td>
<td style="font-weight: 400;">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/regorigregory/FYP_PUBLIC/master?filepath=%2Fexperiments%2Fnotebooks%2FALG_006_EXP_006-Bilateral_sum_truncated_2014-VIS.ipynb)</td>
</tr>
</tbody>
</table>
</ol>
