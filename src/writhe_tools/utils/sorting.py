#!/usr/bin/env python
import numpy as np
import functools
import re
import warnings
import os


def num_str(s, return_str=False, return_num=True, reverse: bool = False):
    if reverse:
        return ''.join(i for i in s if not str.isdigit(i))

    else:
        s = ''.join(filter(str.isdigit, s))
        if return_str and return_num:
            return s, int(s)
        if return_str:
            return s
        if return_num:
            return int(s)


def rm_path(string: str):
    return string.split("/")[-1]


def rm_extension(string: str):
    return ".".join(string.split(".")[:-1])


def lsdir(dir,
          keyword: "list or str" = None,
          exclude: "list or str" = None,
          match: callable = all,
          indexed: bool = False):

    """
        full path version of os.listdir with files/directories in order

        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False
    """

    if dir[-1] == "/":
        dir = dir[:-1]

    listed_dir = filter_strs(os.listdir(dir), keyword=keyword, exclude=exclude, match=match)

    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed)]


def sort_strs(strs: list, max=False, indexed: bool = False):
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function
                    will return normally sorted string list (strs) as an alternative to throwing an error."""

    # we have to ensure that each str in strs contains a number otherwise we get an error
    assert len(strs) > 0, "List of strings is empty"
    check = np.vectorize(lambda s: any(map(str.isdigit, s)))
    if isinstance(strs, list):
        strs = np.array(strs)
    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]
        assert len(strs) > 0, "List of strings is empty after filtering strings without digits"
        indices = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))(strs).argsort()
        return strs[np.flip(indices)] if max else strs[indices]

    # if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on digits that aren't present results in an error
    else:
        if not all(check(strs)):
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                          "If you want to only consider strings that contain a digit, set indexed to True ")
            strs.sort()
            return strs

    indices = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))(strs).argsort()

    return strs[np.flip(indices)] if max else strs[indices]


def filter_strs(strs: list,
                keyword: "list or str" = None,
                exclude: "list or str" = None,
                match: callable = all):
    if keyword is not None:
        strs = keyword_strs(strs, keyword=keyword, exclude=False, match=match)

    if exclude is not None:
        strs = keyword_strs(strs, keyword=exclude, exclude=True, match=match)

    return strs


def keyword_strs(strs: list,
                 keyword: "list or str",
                 exclude: bool = False,
                 match: callable = all,):
    if isinstance(keyword, str):
        filt = (lambda string: keyword not in string) if exclude else\
               (lambda string: keyword in string)
    else:
        filt = (lambda string: match(kw not in string for kw in keyword)) if exclude else\
               (lambda string: match(kw in string for kw in keyword))

    return list(filter(filt, strs))


def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string
    if ignore_case:
        def normalize_old(s):
            return s.lower()
        re_mode = re.IGNORECASE
    else:
        def normalize_old(s):
            return s
        re_mode = 0
    replacements = {normalize_old(key): val for key, val in replacements.items()}
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    pattern = re.compile("|".join(rep_escaped), re_mode)
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)



