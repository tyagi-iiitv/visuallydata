## Featured Projects ##

On this page, we include projects that used our [Visually29K](http://visdata.mit.edu/) dataset for novel applications.
If you would like your project featured on this page, please send an e-mail to zoya@mit.edu with a link to your project repository.
This page is intended to seed new project ideas and give students and researchers some potential starting points for projects.

### Infographics captioning ###

**Goal:** Given an infographic as input, use automated OCR methods to extract the text, and then feed the extracted text to a summarization model to obtain an automatically computed summary or caption for the infographic.
This is challenging because text extracted from infographics is not always correctly parsed, and is often composed of sentence fragments (either due to OCR failures or the structure of the document). 

**Next steps:** Future work can take into account the layout of the infographic document in weighing words for inclusion in the summary. Need to also consider more robust summarization approaches better able to handle the noisy and often fragmented input.

**Demo:** http://visdata.mit.edu/text-attn-viz

**Code:** https://github.com/diviz-mit/text-attn-viz (visualization) and https://github.com/diviz-mit/pointer_gen (summarization)

**Source:** A masters of engineering thesis by [Nathan Landman](https://github.com/landmann): ["Towards abstractive captioning of infographics"](https://dspace.mit.edu/handle/1721.1/119743#files-area)
