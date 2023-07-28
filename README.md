# Compress retrieved collection with DensePhrases

<a href="https://youtu.be/K8NJ71x_Ia8"><img src="https://img.shields.io/badge/Presentation video-FFFFFF?style=for-the-badge&logo=youtube&logoColor=ff0000"/></a> <a href="./assets/docs/[최종] NLP_06조_생성형 검색을 위한 프롬프트 경량화.pdf"><img src="https://img.shields.io/badge/Presentation-FFFFFF?style=for-the-badge&logo=microsoftpowerpoint&logoColor=B7472A"/></a> <a href=""><img src="https://img.shields.io/badge/Wrapup report-FFFFFF?style=for-the-badge&logo=googlesheets&logoColor=34A853"/></a> <a href="https://boostcampait.notion.site/NLP-06-aed368eab95e4b78bcab82d528a18d35?pvs=4"><img src="https://img.shields.io/badge/Project summary-FFFFFF?style=for-the-badge&logo=notion&logoColor=000000"/></a>

## Overview
![Overview](/assets/img/overview.png)

### ODQA(Open-domain Question Answering)
- 질의가 주어지면 주어진 질의에 답할 수 있는 문장들을 Retriever가 지식 베이스로부터 찾아 근거 문서를 구성하고, 구성된 근거 문서(source document)를 기반으로 Reader가 답변하는 시스템

### DensePhrases
- DensePhrases는 질의에 관련된 phrase를 찾는 Retrieval model입니다. 기존의 Dense retrieval model인 문서 전체를 하나의 벡터로 임베딩하는 DPR과 달리, DensePhrases는 phrase 단위로 임베딩 하기 때문에 더 적은 길이의 문서를 출력 할 수 있습니다.

### Project Goal
- mAR 증가를 통한 검색 최적화
- Reader에게 전달되는 근거 문서의 길이를 최소화 하면서 정답 포함율을 유지함으로써 정답 생성에 필요한 비용을 절감합니다.

## Members
|<img src='https://avatars.githubusercontent.com/u/74442786?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/99644139?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/50359820?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/85860941?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/106165619?v=4' height=100 width=100px></img>|
|:---:|:---:|:---:|:---:|:---:|
| [김민호](https://github.com/GrapeDiget) | [김성은](https://github.com/seongeun-k) | [김지현](https://github.com/jihyeeon) | [서가은](https://github.com/gaeun0112) | [홍영훈](https://github.com/MostlyFor) |

### Contribution
- 김민호 : Query loss의 단위 변경, Loss의 구성 요소 추가
- 김성은 : Query loss의 단위 변경, Dynamic Retrieval
- 김지현 : Dataset 전처리, Query loss의 단위 변경, Knowledge Distillation
- 서가은 : Query loss의 단위 변경, Dynamic Retrieval
- 홍영훈 : Query loss의 단위 변경, Static Retrieval, Optimization

## How to run

- [설치 및 실행 방법](/setup.md)

## Demo
![Demo](/assets/img/demo%20page.gif)

## Detail

- [발표 및 시연 영상]()
- [발표 자료]()
- [프로젝트 랩업 리포트]()
- [프로젝트 소개 노션 페이지](https://boostcampait.notion.site/NLP-06-aed368eab95e4b78bcab82d528a18d35?pvs=4)

## Acknowledgement
* Majority of retrieval code comes from [princeton-nlp/Densephrases](https://github.com/princeton-nlp/DensePhrases) and included as submodule of this repository.
* Retrieval-augmented LM is built based on [langchain](https://github.com/hwchase17/langchain).