# Visually29K: a large-scale curated infographics dataset

In this repo, we provide metadata, annotations, and processing scripts for tens of thousands of infographics, for computer vision and natural language research. What kinds of applications can this data be used for? Category (topic) prediction, tagging, popularity prediction (likes & shares), text understanding and summarization, captioning, icon and object detection, design summarization and retargeting. We provide starter code for a subset of these applications, and provide metadata including text detections, icon detections, tag and category annotations available in different formats to make this data easy to use and adaptable to different tasks.

This code is associated with the following project page: http://visdata.mit.edu/ and the manuscripts: ["Synthetically Trained Icon Proposals for Parsing and Summarizing Infographics"](https://arxiv.org/pdf/1807.10441) and ["Understanding infographics through textual and visual tag prediction"](https://arxiv.org/pdf/1709.09215).

In this repository, you will find a number of starter files and scripts, including:

* [howto.ipynb](https://github.com/cvzoya/visuallydata/blob/master/howto.ipynb) shows how to parse the metadata for the infographics. Note that we do not provide the infographics themselves, as they are property of [Visual.ly](https://visual.ly/view), but we do provide URLs for the infographics and a way to obtain them. We also provide the train/test splits which we used for category and tag prediction. The metadata contains attributes that we did not use for our prediction tasks, including popularity indicators (likes & shares), and designer-provided titles and captions. We provide this rich data source to the research community in hope it will spawn future research directions!
* [plot_text_detections.ipynb](https://github.com/diviz-mit/visuallydata/blob/master/plot_text_detections.ipynb) plots detected and parsed text (via Google's OCR API) on top of the infographics, and demonstrates the few different formats we make available from which the parsed text data can be loaded. This text can be a rich resource for natural language processing tasks like captioning and summarization.
* [plot_icon_detections.ipynb](https://github.com/diviz-mit/visuallydata/blob/master/plot_icon_detections.ipynb) loads in our automatically-computed icon detections and classifications for 63K infographics (note that for reporting purposes, only the results on the test set of the 29K subset of infographics are used). These detections and classifications can either be used as a baseline to improve upon, or be used directly as input to new tasks like captioning, retargeting, or summarization.
* [plot_human_annotations.ipynb](https://github.com/diviz-mit/visuallydata/blob/master/plot_human_annotations.ipynb) loads in data for 1.4K infographics that we collected using crowdsourced (Amazon's Mechanical Turk) annotation tasks. Specifically, we asked participants to annotate the locations of icons inside the infographics. Additionally, [human_annotation_consistency.ipynb](https://github.com/diviz-mit/visuallydata/blob/master/human_annotation_consistency.ipynb) provides some scripts for computing consistency between participants at this annotation task. This data is meant to be used as a ground truth for evaluation of computational models.
* [save_tag_to_relevant_infographics.ipynb](https://github.com/diviz-mit/visuallydata/blob/master/save_tag_to_relevant_infographics.ipynb) contains scripts to find and plot the infographics that match different text queries, for a [demo search application](http://visdata.mit.edu/explore.html). Search engines typically use meta-data to determine which images to serve based on a search query. They do not look inside the image. Our automatically pre-computed detections allow us to find the infographics that contain matching text and icons.

Links to large data files that could not be directly stored in this repository can be found in [links_to_data_files.md](links_to_data_files.md)

If you use the data or code in this git repo, please consider citing:
``` 
@inproceedings{visually2,
    author    = {Spandan Madan*, Zoya Bylinskii*, Matthew Tancik*, Adrià Recasens, Kimberli Zhong, Sami Alsheikh, Hanspeter Pfister, Aude Oliva, Fredo Durand}
    title     = {Synthetically Trained Icon Proposals for Parsing and Summarizing Infographics},
    booktitle = {arXiv preprint arXiv:1807.10441},
    url       = {https://arxiv.org/pdf/1807.10441},
    year      = {2018}
}
@inproceedings{visually1,
    author    = {Zoya Bylinskii*, Sami Alsheikh*, Spandan Madan*, Adria Recasens*, Kimberli Zhong, Hanspeter Pfister, Fredo Durand, Aude Oliva}
    title     = {Understanding infographics through textual and visual tag prediction},
    booktitle = {arXiv preprint arXiv:1709.09215},
    url       = {https://arxiv.org/pdf/1709.09215},
    year      = {2017}
}
```


