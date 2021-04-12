import re


class BaseFilter(object):
    def __init__(self):
        self.patterns = []
        self.init_patterns()

    def init_patterns(self):
        pass

    def filter(self, content):
        for t in self.patterns:
            m = re.search(t, content)
            if m is not None:
                return True
        return False

    @staticmethod
    def omit_garbage_information(content):
        paragraphs = content.split('\n\n')
        res = ''
        for p in paragraphs:
            lines = p.split('\n')
            if len(lines) > 30:
                continue
            res += p
            res += '\n\n'
        if len(res) > 2:
            res = res[:-2]
        return res

    def omit_structural_information(self, content):
        for t in self.patterns:
            content = re.sub(t, ' ', content)
        return content


class ScreenshotFilter:
    def filter(self, content):
        if content is None:
            return False
        content = content.lower()
        af = AttachmentFilter()
        lines = content.split('\n')
        attachment = False
        for l in lines:
            if attachment:
                for word in ['window', 'view', 'picture', 'screenshot', 'visible',
                             'image', 'png', 'bmp', 'jpg', 'jpeg', 'where to',
                             'screen shot','yellow', 'rectangle', 'snapshot']:
                    if word in l:
                        return True
                attachment = False
            if af.filter(l):
                attachment = True
        return False


class TestCaseFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'test case[s]?:',
            r'testcase[s]?:',
            r'test case[s]?[\s]+\(.*\):'
            r'testcase[s]?[\s]+\(.*\):'
            r'^test case[s]?[\s]*\n',
            r'^testcase[s]?[\s]*\n',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE | re.MULTILINE)
            self.patterns.append(p)

    def filter(self, content):
        if content is None:
            return False
        content = content.lower()
        af = AttachmentFilter()
        lines = content.split('\n')
        attachment = False
        for l in lines:
            if attachment:
                for word in ['test case', 'testcase', 'added test', 'test program', 'testing case']:
                    if word in l:
                        return True
                attachment = False
            if af.filter(l):
                attachment = True
        for t in self.patterns:
            m = re.search(t, content)
            if m is not None:
                return True
        return False


class AttachmentFilter(BaseFilter):
    def init_patterns(self):
        template = r'Created attachment [0-9]+'
        p = re.compile(template, re.IGNORECASE)
        self.patterns.append(p)


class PatchFilter(BaseFilter):
    def init_patterns(self):
        template = r'[-]{3}[\s].*\n[\+]{3}[\s].*\n[@]{2}'
        p = re.compile(template, re.IGNORECASE)
        self.patterns.append(p)

    def filter(self, content):
        if content is None:
            return False
        content = content.lower()
        af = AttachmentFilter()
        lines = content.split('\n')
        attachment = False
        for l in lines:
            if attachment:
                for word in ['fix', 'patch']:
                    if word in l:
                        return True
                attachment = False
            if af.filter(l):
                attachment = True
        for t in self.patterns:
            m = re.search(t, content)
            if m is not None:
                return True
        return False

    def omit_structural_information(self, content):
        paragraphs = content.split('\n\n')
        res = ''
        i = 0
        while not self.filter(paragraphs[i]):
            res += paragraphs[i]
            res += '\n\n'
            i += 1
        if len(res) > 2:
            res = res[:-2]
        return res


class JavaStackTraceFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'^\!SUBENTRY .*',
            r'^\!ENTRY .*',
            r'^\!MESSAGE .*',
            r'^\!STACK .*',
            r'^[\s]*at[\s]+.*[\n]?\([\w]+\.java(:[\d]+)?\)',
            r'^[\s]*([\w]+\.)+[\w]+(Exception|Error)(:[\s]+(.*\n)*.*)?',
            # r'^[\s]*at[\s]+([\w]+[\$}?[\w]*\.)+[\w]+[\$]?[\w]*[\n]?\([\w]+\.java(:[\d]+)?\)',
            # r'^[\s]*at[\s]+([\w]+[\$]?[\w]*\.)+\<[\w]+[\$]?[  \w]*\>[\n]?\([\w]+\.java(:[\d]+)?\)',
        ]
        for t in templates:
            p = re.compile(t, re.MULTILINE)
            self.patterns.append(p)

    def filter(self, content):
        if content is None:
            return False
        content = content.lower()
        af = AttachmentFilter()
        lines = content.split('\n')
        attachment = False
        for l in lines:
            if attachment:
                for word in ['trace']:
                    if word in l:
                        return True
                attachment = False
            if af.filter(l):
                attachment = True
        for t in self.patterns:
            m = re.search(t, content)
            if m is not None:
                return True
        return False

    def omit_structural_information(self, content):
        paragraphs = content.split('\n\n')
        res = ''
        for p in paragraphs:
            if self.filter(p):
                continue
            res += p
            res += '\n\n'
        if len(res) > 2:
            res = res[:-2]
        return res


class GdbStackTraceFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'#[\d]+[\s]+0x[0-9a-f]{16}[\s]+in[\s]+[\S]+',
            r'Thread[\s]+[\d]+[\s]+\(process[\s]+[\d]+\):\n#[\d]+'
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE)
            self.patterns.append(p)


class JavaScriptStackTraceFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'^[\s]*[\S]+@[\S]+\.js:[\d]+',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)


class DmesgFilter(BaseFilter):
    def init_patterns(self):
        template = r'[\s]*\[[\s]*[\d]+\.[\d]+\]'
        p = re.compile(template, re.IGNORECASE)
        self.patterns.append(p)


class ReproduceStepFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'step[s]? to reproduce',
            r'reproduce step[s]?',
            r'step[s]? for reproduction',
            r're-creation proc',
            r'repro[\w]*[\s]+step',
            r'to reproduce',
            r'reproducible: always',
            r'repro[\w]*frequency',
            r'step[s]?',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE)
            self.patterns.append(p)
        p = re.compile(r'^[\W]*STEPS[\W]*$', re.MULTILINE)
        self.patterns.append(p)
        p = re.compile(r'^[\W]*STR:[\W]*$', re.MULTILINE)
        self.patterns.append(p)


class CodeFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'^[\s]*(public|private|protected).*class[\s]+[\w]+[\s]',
            r'^[\s]*(public|private|protected).*\(.*\)[\n]?',
            r'^[\s]*(if|for|while)[\s]*\(.*\)',
            r'\{(.*\n)*.*\}'
            r'import[\s]+.*;'
            # r'\/\*.*(\n.*)*\*\/',
            # r'^[\s]+.*;([\s]+\/\/.*)?\n',
            # r'^[\s]*\/\/.*\n',
            # r'[\w]+\.[\w]+(\.[\w]+)*[\s]*\(.*\)',
            # r'\}',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)


class ConfigurationFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'jvm properties:[\n]?\{.*\}',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)


class JavaThreadDumpFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'^thread[\s]+[\d]+([\s]+[\w]+)?:',
            r'#[\d]+[\s]+0x[0-9a-f]{8}[\s]+in.*\n',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)

    def omit_structural_information(self, content):
        paragraphs = content.split('\n\n')
        res = ''
        for p in paragraphs:
            if self.filter(p):
                continue
            res += p
            res += '\n\n'
        if len(res) > 2:
            res = res[:-2]
        return res
    
class ResultFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'actual result[s]?',
            r'expected result[s]?',
            r'actual',
            r'expected',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)
    
class EnvironmentFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'build identifier',
            r'user[-]?agent',
            r'environmental variable[s]?',
            r'java version',
            r'jvm properties',
            r'svn client',
            r'build:',
            r'vm:',
            r'os:',
            r'product version:',
            r'operating system:',
            r'java:',
            r'configuration detail[s]?',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)
            
class CommentsFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'user comment[s]?',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)
            
class ErrorFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'error detail[s]?',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)
            
class OtherFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'automated alert report from',
            r'legal message',
            r'3rd party materials',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)
            
class LogFilter(BaseFilter):
    def init_patterns(self):
        templates = [
            r'error log',
            r'ide log',
            r'server log',
        ]
        for t in templates:
            p = re.compile(t, re.IGNORECASE|re.MULTILINE)
            self.patterns.append(p)

