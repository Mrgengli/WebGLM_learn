import re, os
from rouge_score import rouge_scorer, tokenize

class DataUtils:
    @staticmethod
    def split_segments(statement: str):
        # 该函数的输入是一个字符串 statement，它表示待处理的文本语句。函数将文本进行分割和解析，返回一个包含所有段落的列表和相应的引用列表。
        # 具体来说，函数首先去除多余的空格和换行符，并使用正则表达式 split_pattern 对语句进行分割。
        # 然后，根据语句中的引用标记（以方括号表示），将语句划分为段落和引用列表的组合。
        # 接下来，将每个段落进一步分割成较短的句子，同时将引用列表添加到相应的段落中。最后，函数返回经过处理的段落列表和引用列表。
        all_statements = []
        statement = re.sub(' +', ' ', statement.replace('\n', ' '))
        split_pattern = r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*(?:\W*)([A-Z])'
        tmp_statements = []
        
        for s in re.split(r"(\[\d+\])", statement):
            if not s:
                continue
            cites = re.findall(r"\[(\d+)\]", s)
            if not cites: # Segment
                tmp_statements.append([s, []])
            elif not tmp_statements: # Citation Mark, but no Segments
                continue
            else: # Citation Mark
                for item in cites:
                    tmp_statements[-1][1].append(int(item) - 1)
        
        for s, cite in tmp_statements:
            prefix = ""
            for ix, seg in enumerate(re.split(split_pattern, s)):
                if len(prefix) > 20:
                    all_statements.append([prefix, []])
                    prefix = ""
                prefix += seg
                if prefix and prefix[-1] in ['.!?:']:
                    prefix += " "
            if prefix:
                if all_statements and len(prefix) < 20:
                    all_statements[-1][0] += prefix
                else:
                    all_statements.append([prefix, []])
            if all_statements:
                all_statements[-1][1] += cite
        
        return [seg[0] for seg in all_statements], [seg[1] for seg in all_statements]
    
    @staticmethod
    def matching_score(all_statements, references):
        # 该函数的输入是两个列表 all_statements 和 references，分别表示所有段落和相应的引用。
        # 该函数首先定义了一个内部函数 remove_stopwords，用于去除段落和引用中的停用词。
        # 然后，函数通过遍历所有段落和引用，并调用 remove_stopwords 函数来去除停用词。
        # 接下来，函数创建了一个 RougeScorer 对象，只计算 Rouge-1 分数。然后，函数开始计算段落与引用之间的匹配得分。
        # 对于每个段落，如果单词数少于5个，则将其与所有引用之间的得分都设为0。
        # 否则，函数使用 Rouge-1 精确度计算当前段落与每个引用之间的得分，并将它们添加到 all_scores 列表中。
        # 最后，函数返回包含所有段落与引用之间得分的二维列表 all_scores。
        # 请注意，该函数依赖于一些其他模块（例如 tokenize 和 rouge_scorer），在使用之前需要先导入这些模块。
        # 另外，函数中涉及的停用词处理和 Rouge-1 得分计算可能需要额外的资源或库支持。
        def remove_stopwords(stmt: str):
            stmt = tokenize.tokenize(stmt, None)
            ret = []
            for item in stmt:
                if item in stopwords:
                    continue
                ret.append(item)
            return " ".join(ret)
        
        all_statements = [remove_stopwords(item) for item in all_statements]
        references = [remove_stopwords(item) for item in references]
        
        # return None
        scorer = rouge_scorer.RougeScorer(['rouge1'])
        all_scores = []
        for statement in all_statements:
            if len(tokenize.tokenize(statement, None)) < 5:
                all_scores.append([0] * len(references))
                continue
            ref_score = []
            for idx, ref in enumerate(references):
                rouge = scorer.score(ref, statement)['rouge1'].precision
                # print(rouge)
                ref_score.append(rouge)
            all_scores.append(ref_score)
        return all_scores
    
    @staticmethod
    def get_ideal_citations(all_scores, raw_citations, citation_threshold, extra_bonus=0.3):
        # 该静态方法的输入包括三个列表 all_scores、raw_citations 和一个引用阈值 citation_threshold，以及一个额外奖励分数 extra_bonus（默认为0.3）。方法首先使用断言来确保得分列表和原始引用列表的长度相等。
        # 然后，方法遍历所有段落的得分列表，并根据得分和一些条件进行判断。对于每个得分，如果其对应的引用在原始引用列表中，则给它增加额外的奖励分数。然后，如果得分超过引用阈值，则将其添加到理想引用列表中。同时，方法会记录最佳得分和对应的引用索引。
        # 在遍历完所有得分之后，如果理想引用列表为空且原始引用列表不为空，则将最佳引用作为理想引用。最后，方法返回包含所有段落的理想引用列表 ideal_citations。
        # 请注意，该方法是一个静态方法，可以通过类名直接调用，而无需创建类的实例。
        assert len(all_scores) == len(raw_citations)
        
        ideal_citations = []
        for seg_idx, scores in enumerate(all_scores):
            idc = []
            best_idx = 0
            best_scr = 0
            for idx, score in enumerate(scores):
                if idx in raw_citations[seg_idx]:
                    score += extra_bonus / len(raw_citations[seg_idx])
                if score >= citation_threshold:
                    idc.append(idx)
                if score > best_scr:
                    best_idx = idx
            if len(idc) == 0 and len(raw_citations[seg_idx]) > 0:
                idc.append(best_idx)
            ideal_citations.append(idc)
        return ideal_citations
    
    @staticmethod
    def recompose(all_statements, raw_citations, references, sep=" ", citation_threshold=0.75) -> str:
        scores = DataUtils.matching_score(all_statements, references)
        ret = ""
        ideal_citations = DataUtils.get_ideal_citations(scores, raw_citations, citation_threshold)
        for seg, cit in zip(all_statements, ideal_citations):
            # judge if seg[0] is alphanumeric
            if ret and ret[-1] == "]" and seg and seg[0].isalnum():
                ret += sep
            ret += seg
            for c in cit:
                ret += "[%d]"%(c+1)
            if ret and ret[-1] in ".!?:":
                ret += sep
        return ret.strip()

class Stopwords:
    @staticmethod
    def load():
        src = [
            "./model/stopwords/english",
            "./model/stopwords/explaination",
        ]
        ret = []
        for item in src:
            with open(item, "r") as f:
                ret += [word.strip() for word in f.readlines()]
        return ret


stopwords = set(Stopwords.load())

def citation_correction(original_answer, references):
    segments, raw_cite = DataUtils.split_segments(original_answer)
    
    return DataUtils.recompose(segments, raw_cite, references)
