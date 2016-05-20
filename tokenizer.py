import re
class Token(object):
    PTN = re.compile('.*')
    def __init__(self):
        self.content = ''
        self.subtokens = {}
        self.comment = ''

    @classmethod
    def test(cls, content):
        return re.match(cls.PTN, content)

    @classmethod
    def generate(cls, content):
        content = filter(lambda x: not x in [' ', '\r', '\t', '\n'], content)
        if '//' in content:
            content = content.split('//')
            comment = '//'.join(content[1:])
            content = content[0]
        else:
            comment = ''

        token = cls()
        token.content = content
        token.comment = comment
        match = re.match(cls.PTN, content)
        if not match:
            return None
        for item in match.groupdict():
            if match.group(item):
                key, token_type = item.split('_')
                token.subtokens[key] = eval("{}.generate(\'\'\'{} \'\'\')".format(token_type, match.group(item)))
        return token

    def feature_extraction(self):
        self.feature = [(type(self).__name__, self.content)]
        for key in self.subtokens:
            token = self.subtokens[key]
            if token:
                self.feature += token.feature_extraction()
        return self.feature

class Assignment(Token):
    PTN = re.compile('^(?P<left_Variable>.*)=(?P<right_Expression>.*)$')

class IfStmt(Token):
    PTN = re.compile('^if\((?P<cond_Expression>.*)\)(?P<body_Expression>.*)$')

class WhileStmt(Token):
    PTN = re.compile('^while\((?P<cond_Expression>.*)\)(?P<body_Expression>.*)$')

class ForStmt(Token):
    PTN = re.compile('^for\((?P<init_Expression>.*);(?P<cond_Expression>.*);(?P<end_Expression>.*)\)(?P<body_Expression>.*)$')

class MethodCall(Token):
    PTN = re.compile('^(?P<method_Variable>[^\.]+)\.(?P<second_SecondaryMethodCall>.*)$')

class SecondaryMethodCall(Token):
    PTN = re.compile('^(?P<method_Variable>[^\.]+)\.?(?P<second_SecondaryMethodCall>.*)$')

class Expression(Token):
    @classmethod
    def generate(cls, content):
        token = cls()
        token.content = content
        return token

class Variable(Token):
    @classmethod
    def generate(cls, content):
        token = cls()
        token.content = content
        return token

TOKEN_TABLE = [WhileStmt, ForStmt, IfStmt, Assignment, MethodCall, Token]

def Tokenize(code):
    for cls in TOKEN_TABLE:
        if cls.test(code):
            return cls.generate(code)
    return
