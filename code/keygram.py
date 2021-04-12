from keygramFilters import JavaStackTraceFilter
from keygramFilters import GdbStackTraceFilter
from keygramFilters import JavaScriptStackTraceFilter
from keygramFilters import ReproduceStepFilter
from keygramFilters import PatchFilter
from keygramFilters import CodeFilter
from keygramFilters import AttachmentFilter
from keygramFilters import ScreenshotFilter
from keygramFilters import TestCaseFilter
from keygramFilters import EnvironmentFilter
from keygramFilters import ResultFilter
from keygramFilters import CommentsFilter
from keygramFilters import ErrorFilter
from keygramFilters import OtherFilter
from keygramFilters import LogFilter

filters = dict()


def get_filters():
    global filters
    filters['stack_trace'] = [JavaStackTraceFilter(), GdbStackTraceFilter(), JavaScriptStackTraceFilter()]
    filters['step_reproduce'] = [ReproduceStepFilter()]
    filters['patch'] = [PatchFilter()]
    filters['code'] = [CodeFilter()]
    filters['attachment'] = [AttachmentFilter()]
    filters['screenshot'] = [ScreenshotFilter()]
    filters['testcase'] = [TestCaseFilter()]
    filters['environment'] = [EnvironmentFilter()]
    filters['result'] = [ResultFilter()]
    filters['comments'] = [CommentsFilter()]
    filters['error'] = [ErrorFilter()]
    filters['other'] = [OtherFilter()]
    filters['log'] = [LogFilter()]


class Keygram:
    def __init__(self, text):
        self.desc=text
        
    def extract(self):
        get_filters()
        keyg = dict()
        if self.desc is None:
            for filter_name in filters.keys():
                keyg['has_'+filter_name] = 0
            return keyg
        for filter_name in filters.keys():
            this_filter = filters[filter_name]
            feature_name = 'has_' + filter_name
            keyg[feature_name] = 0
            if len(this_filter) == 0:
                keyg[feature_name] = this_filter[0].filter(self.desc)
            else:
                for f in this_filter:
                    if f.filter(self.desc):
                        keyg[feature_name] = 1
        if keyg['has_patch']:
            keyg['has_code'] = 0
        return keyg
