---
layout: index
title: "About"
---


# About this page.

This page contains a curated list of AI for EDA papers.
It is under construction and you are welcomed to submit your publications.
This page can **automatically** render the plain `bibtex` file into `html` for display on the webpage.
So you only need to submit your bibtex file to [https://github.com/ai4eda/awesome-AI4EDA](https://github.com/ai4eda/awesome-AI4EDA).
Or contact [Guojin Chen](https://gjchen.me) by email [cgjcuhk@gmail.com](cgjcuhk@gmail.com)


## Example

![Bibtex2Html](/images/bibtex2html.png)


<br/>

## How to contribute / add my publications?

### Step 1: Add your `bibtex` file to `./publications/***.bib`

We provide different categories according to the EDA flow, please copy your bibtex to the corresponding category.

The categories are defined in `pub.yaml` of repo: [awesome-AI4EDA](https://github.com/ai4eda/awesome-AI4EDA), you can also submit new categories to `pub.yaml` or contact [cgjcuhk@gmail.com](mailto:cgjcuhk@gmail.com) to help you add the categories. 

```yaml
# pub.yaml
categories_publications:
  name: "All publications"
  group_by_topic: True
  categories:
    -
      heading: "High Level Synthesis"
      file: hls.bib
      prefix: ''
    -
      heading: "Logic Synthesis"
      file: logicSyn.bib
      prefix: ''
    -
      heading: "Circuit Verification"
      file: circuit.bib
      prefix: ''
    -
      heading: "Placement"
      file: place.bib
      prefix: ''
    -
      heading: "Clock Tree Synthesis"
      file: CTS.bib
      prefix: ''
    -
      heading: "Routing"
      file: routing.bib
      prefix: ''
    -
      heading: "Layout Verification"
      file: layoutVer.bib
      prefix: ''
    -
      heading: "Mask Optimization"
      file: opc.bib
      prefix: ''
```



### Step 2: Add the topic for your pub in the bibtex file.

Example: 👇🏻 the `_venue`, `year`, `topic` fileds are required. You publication will be displayed into the corresponding topic.

You can also add `url` filed to attach the paper link. 
And the `abstract` filed to add the paper abstract.
Add the `code` filed for link to your codes.

```bibtex
@inproceedings{DAC23_Nitho,
  title={Physics-Informed Optical Kernel Regression Using Complex-valued Neural Fields},
  author={Chen, Guojin and Pei, Zehua and Yang, Haoyu and Ma, Yuzhe and Yu, Bei and Wong, Martin},
  booktitle={ACM/IEEE Design Automation Conference,  (\textbf{DAC '23})},
  _venue={DAC},
  year={2023},
  topic = {Lithography},
  url = {link to your paper},
  abstract = {abstract of your paper},
  code = {link to your codes}
}
```

### Step3: Submit a PR  or email to Guojin Chen ([cgjcuhk@gmail.com](mailto:cgjcuhk@gmail.com))

Submit a PR to repo: [awesome-AI4EDA](https://github.com/ai4eda/awesome-AI4EDA)

Thank you.


## Acknowledgement

Thanks for the contribution and support from Prof. [Bei Yu](https://www.cse.cuhk.edu.hk/~byu/), Prof. [Yibo Lin](https://yibolin.com/) and Mr.[Jing Mai](https://magic3007.github.io/)


<br/>
<br/>
<br/>