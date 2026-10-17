from collections import OrderedDict
import xml.etree.ElementTree as ET

__all__ = ["XMLParser",]

TRUE_FALSE_NONE = {"true": True, "false": False, "none": None}


def _parse_label_value(value: str):
    if value is None:
        return value
    value = value.strip()
    if not value:
        return False
    return TRUE_FALSE_NONE.get(value.lower(), value)


def _process_general_labels(node_testcase):
    ret = {
        "disable": False,
    }
    labels = node_testcase.find('labels')
    if labels is not None:
        for key, value in labels.attrib.items():
            ret[key] = _parse_label_value(value)
    return ret


def _process_build_parameters(node_testcase):
    b = node_testcase.find('build')
    pars_tfn_default = {
        "require_libri": "none",
    }
    ret = {}
    for p, de in pars_tfn_default.items():
        try:
            value = b.get(p, de)
        except AttributeError:
            value = de
        ret[p] = TRUE_FALSE_NONE.get(value.lower(), value)
    return ret


def _process_run_parameters(node_testcase):
    r = node_testcase.find('run')
    ret = {}

    pars_intrange = {
        "ntasks_disable": "none",
        "nthreads_disable": "none",
        "ntasks_enable": "none",
        "nthreads_enable": "none",
    }
    for p, de in pars_intrange.items():
        try:
            try:
                value = r.get(p, de)
            except AttributeError:
                value = de
            if value.lower() in ["none", "false"]:
                value = tuple()
            else:
                value = tuple(int(x) for x in value.replace(',', ' ').split())
        except ValueError:
            value = None
        ret[p] = value
    try:
        copy_files = r.get("copy_files", "")
    except AttributeError:
        copy_files = ""
    if copy_files.lower() in ["none", "false"]:
        ret["copy_files"] = tuple()
    else:
        ret["copy_files"] = tuple(x for x in copy_files.replace(',', ' ').split())
    return ret


def _process_validate_parameters(node_validate):
    nv = node_validate
    # name is manditory
    name = nv.get('name')
    file = nv.get('file', 'librpa/librpa.out')
    file_test = nv.get('file_test', file)
    file_refr = nv.get('file_refr', file)

    pars = ["regex", "occurences", "headers", "rows", "comparison", "binary_extract",
            "binary_extract_preset"]
    ret = dict(name=name, file=file, file_test=file_test, file_refr=file_refr)
    for p in pars:
        ret[p] = nv.get(p, None)

    collective = nv.get("collective", "false").strip().lower()
    if collective not in ("true", "false"):
        raise ValueError("collective must be true or false")
    ret["collective"] = TRUE_FALSE_NONE[collective]

    return ret


class XMLParser:

    def __init__(self, path_xml: str):
        tree = ET.parse(path_xml)
        root = tree.getroot()
        groups = OrderedDict()
        for group in root.findall('group'):
            gname = group.get('name')
            if gname not in groups:
                groups[gname] = []
            for tc in group.findall('testcase'):
                name = tc.get('name')
                directory = tc.get('directory')
                build = _process_build_parameters(tc)
                run = _process_run_parameters(tc)
                labels = _process_general_labels(tc)
                validates = []
                for v in tc.findall('validate'):
                    validates.append(_process_validate_parameters(v))
                groups[gname].append(
                    dict(name=name, directory=directory, build=build, run=run,
                         validates=validates, labels=labels)
                )
        self.groups = groups
