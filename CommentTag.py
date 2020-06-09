import json
class CommentTag(object):

    # 过滤得到负向标签, 得到句子
    @staticmethod
    def get_neg_info(json_list: list):
        needs = ["sort", "topic", "clause"]
        clause = map(lambda y: {key: y.get(key) for key in needs}, filter(lambda x: x.get("emotionType", None) == 0, json_list))
        return list(clause)

    @staticmethod
    def parse_comment_tag(json_str: str):
        if not json_str:
            return "Error: None"
        try:
            json_objs = json.loads(json_str)
        except:
            return "Error: parse"

        return CommentTag.get_neg_info(json_objs.get("tags", []))

    @staticmethod
    def combin_id(comment_no, tag_json_str):
        neg_list = CommentTag.parse_comment_tag(tag_json_str)
        info = "\t".join(list(map(lambda x: json.dumps(x), neg_list)))
        return "{comment_no}#{info}".format(comment_no=comment_no, info=info)

    @staticmethod
    def sql_filter():
        sql = """
        select 
            comment_no,  comment_tag
        from ods.ods_comment_data_comment where comment_tag is not null and comment_tag != 'null' 
        """
        return sql






