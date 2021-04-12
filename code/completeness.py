from completenessFilters import JavaStackTraceFilter
from completenessFilters import GdbStackTraceFilter
from completenessFilters import JavaScriptStackTraceFilter
from completenessFilters import ReproduceStepFilter
from completenessFilters import PatchFilter
from completenessFilters import CodeFilter
from completenessFilters import AttachmentFilter
from completenessFilters import ScreenshotFilter
from completenessFilters import TestCaseFilter

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


class Completeness:
    def __init__(self, text):
        self.desc=text
        
    def extract(self):
        get_filters()
        compl = dict()
        if self.desc is None:
            for filter_name in filters.keys():
                compl['has_'+filter_name] = 0
            return compl
        for filter_name in filters.keys():
            this_filter = filters[filter_name]
            feature_name = 'has_' + filter_name
            compl[feature_name] = 0
            if len(this_filter) == 0:
                compl[feature_name] = this_filter[0].filter(self.desc)
            else:
                for f in this_filter:
                    if f.filter(self.desc):
                        compl[feature_name] = 1
        if compl['has_patch']:
            compl['has_code'] = 0
        return compl
