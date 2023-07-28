import argparse
import base64
import json
import http.client
import sys
import gradio as gr

from retrieve import retriever

DEFAULT_QUESTION = "Wikipedia 2018 english dump에서 궁금한 점을 질문해주세요.\n예를들어 \n\n- Where are mucosal associated lymphoid tissues present in the human body and why?\n- When did korean drama started in the philippines?\n- When did the financial crisis in greece start?"


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/completions/LE-C',
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['text']
        else:
            return 'Error'


if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiY1/n/zxXKzrnSqsIABCAJI/8nsOf4fNMPgIT/+pTJR0Irvr6HGSZNLIQdr/vIDtDmFUyUTv1Rt7shp//y5N2hdJ/Ku7zVxBiBUhtOI/K/LcpAt5aEsNg2s/5DpHVH3VS09R50V1CLYc8T49hvZrRt6ym7jqoU9rCFd06TZyzk9V0Lj6cBFjq/0iPVACVn9/yEDo5j7EUDlrZ1UGFRjX3i3c=',
        api_key_primary_val='tJEDn2jqkk3Sy3ph0LDEZiB3hvKaEgtwugzlh3cu',
        request_id='7f1db3378543404fb04d537cd183a93c'
    )

    question = 'when is the next deadpool movie being released'
    summaries = """Content: A sequel, "Deadpool 2", was released in May 2018.
Source: 2353979
Content: It was released in the United States on May 18, 2018, having been previously scheduled for release on June 1 of that year.
Source: 2353982
Content: Deadpool 2" was released on May 18, 2018, with Baccarin, T. J. Miller, Uggams, Hildebrand, and Kapičić all returning.
Source: 2353979
Content: "Deadpool 2" was released in the United States on May 18, 2018.
Source: 2353982
Content: Also in January, the film's release was moved up to May 18, 2018.
Source: 2353982
Content: An extended edition was released in August 2018, and a re-cut, PG-13 version of the film titled "Once Upon a Deadpool" was released theatrically in December 2018.
Source: 2353982
Content: "Deadpool" was released in the United States on February 12, 2016, after an unconventional marketing campaign.
Source: 2353979
Content: Also in April, Leslie Uggams confirmed that she would be reprising her role of Blind Al from the first film, while Fox gave the sequel a June 1, 2018 release date.
Source: 2353982
Content: That September, Fox gave "Deadpool" a release date of February 12, 2016.
Source: 2353979"""

    preset_text = f"""Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
==
QUESTION: Which state/country's law governs the interpretation of the contract?
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.
Source: 30
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4
==
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28
==
QUESTION: {question}
{summaries}
==
FINAL ANSWER:"""

    request_data = {
        'text': preset_text,
        'maxTokens': 32,
        'temperature': 0.5,
        'topK': 0,
        'topP': 0.8,
        'repeatPenalty': 5.0,
        'start': '',
        'restart': '',
        'stopBefore': [],
        'includeTokens': True,
        'includeAiFilters': True,
        'includeProbs': False
    }

    response_text = completion_executor.execute(request_data)
    # print('*'*50)
    # print(preset_text)
    print('*'*50)
    print(response_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask a question.")

    # General
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-16k-0613",
        help="model name for openai api",
    )

    # Retriever: Densephrase
    parser.add_argument(
        "--query_encoder_name_or_dir",
        type=str,
        default="princeton-nlp/densephrases-multi-query-multi",
        help="query encoder name registered in huggingface model hub OR custom query encoder checkpoint directory",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="start/1048576_flat_OPQ96_small",
        help="index name appended to index directory prefix",
    )

    args = parser.parse_args()

    # to prevent collision with DensePhrase native argparser
    sys.argv = [sys.argv[0]]

    # initialize class
    app = RaLM(args)

    def question_answer(question):
        result = app.run_chain(question=question, force_korean=False)

        return result["answer"], "\n######################################################\n\n".join(
            [
                f"Source {idx}\n{doc.page_content}"
                for idx, doc in enumerate(result["source_documents"])
            ]
        )

    # launch gradio
    gr.Interface(
        fn=question_answer,
        inputs=gr.inputs.Textbox(default=DEFAULT_QUESTION, label="질문"),
        outputs=[
            gr.inputs.Textbox(default="챗봇의 답변을 표시합니다.", label="생성된 답변"),
            gr.inputs.Textbox(
                default="prompt에 사용된 검색 결과들을 표시합니다.", label="prompt에 첨부된 검색 결과들"
            ),
        ],
        title="지식기반 챗봇",
        theme="dark-grass",
        description="사용자의 지식베이스에 기반해서 대화하는 챗봇입니다.\n본 예시에서는 wikipedia dump에서 검색한 후 이를 바탕으로 답변을 생성합니다.\n\n retriever: densePhrase, generator: gpt-3.5-turbo-16k-0613 (API)",
    ).launch(share=True)
