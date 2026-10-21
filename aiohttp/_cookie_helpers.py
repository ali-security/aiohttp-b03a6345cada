"""
Internal cookie handling helpers.

This module contains internal utilities for cookie parsing and manipulation.
These are not part of the public API and may change without notice.
"""

import re
from collections.abc import Sequence
from http.cookies import CookieError, Morsel

from .log import internal_logger

__all__ = (
    "parse_set_cookie_headers",
    "parse_cookie_header",
    "preserve_morsel_with_coded_value",
)

# Cookie parsing constants
# Allow more characters in cookie names to handle real-world cookies
# that don't strictly follow RFC standards (fixes #2683)
# RFC 6265 defines cookie-name token as per RFC 2616 Section 2.2,
# but many servers send cookies with characters like {} [] () etc.
# This makes the cookie parser more tolerant of real-world cookies
# while still providing some validation to catch obviously malformed names.
_COOKIE_NAME_RE = re.compile(r"^[!#$%&\'()*+\-./0-9:<=>?@A-Z\[\]^_`a-z{|}~]+$")
_COOKIE_KNOWN_ATTRS = frozenset(  # AKA Morsel._reserved
    (
        "path",
        "domain",
        "max-age",
        "expires",
        "secure",
        "httponly",
        "samesite",
        "partitioned",
        "version",
        "comment",
    )
)
_COOKIE_BOOL_ATTRS = frozenset(  # AKA Morsel._flags
    ("secure", "httponly", "partitioned")
)

# Implementation limits as permitted by RFC 6265 Section 6.1.
_MAX_COOKIES_PER_RESPONSE = 50
_MAX_COOKIE_PAIR_LENGTH = 4096
_MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH = 1024

_COOKIE_CTL_RE = re.compile(r"[\x00-\x1f\x7f]")

# SimpleCookie's pattern for parsing cookies with relaxed validation
# Based on http.cookies pattern but extended to allow more characters in cookie names
# to handle real-world cookies (fixes #2683)
_COOKIE_PATTERN = re.compile(
    r"""
    \s*                            # Optional whitespace at start of cookie
    (?P<key>                       # Start of group 'key'
    # aiohttp has extended to include [] for compatibility with real-world cookies
    [\w\d!#%&'~_`><@,:/\$\*\+\-\.\^\|\)\(\?\}\{\[\]]+   # Any word of at least one letter
    )                              # End of group 'key'
    (                              # Optional group: there may not be a value.
    \s*=\s*                          # Equal Sign
    (?P<val>                         # Start of group 'val'
    "(?:[^\\"]|\\.)*"                  # Any double-quoted string (properly closed)
    |                                  # or
    "[^";]*                            # Unmatched opening quote (differs from SimpleCookie - issue #7993)
    |                                  # or
    # Special case for "expires" attr - RFC 822, RFC 850, RFC 1036, RFC 1123
    (\w{3,6}day|\w{3}),\s              # Day of the week or abbreviated day (with comma)
    [\w\d\s-]{9,11}\s[\d:]{8}\s        # Date and time in specific format
    (GMT|[+-]\d{4})                     # Timezone: GMT or RFC 2822 offset like -0000, +0100
                                        # NOTE: RFC 2822 timezone support is an aiohttp extension
                                        # for issue #4493 - SimpleCookie does NOT support this
    |                                  # or
    # ANSI C asctime() format: "Wed Jun  9 10:18:14 2021"
    # NOTE: This is an aiohttp extension for issue #4327 - SimpleCookie does NOT support this format
    \w{3}\s+\w{3}\s+[\s\d]\d\s+\d{2}:\d{2}:\d{2}\s+\d{4}
    |                                  # or
    [\w\d!#%&'~_`><@,:/\$\*\+\-\.\^\|\)\(\?\}\{\=\[\]]*      # Any word or empty string
    )                                # End of group 'val'
    )?                             # End of optional value group
    \s*                            # Any number of spaces.
    (\s+|;|$)                      # Ending either at space, semicolon, or EOS.
    """,
    re.VERBOSE | re.ASCII,
)


def preserve_morsel_with_coded_value(cookie: Morsel[str]) -> Morsel[str]:
    """
    Preserve a Morsel's coded_value exactly as received from the server.

    This function ensures that cookie encoding is preserved exactly as sent by
    the server, which is critical for compatibility with old servers that have
    strict requirements about cookie formats.

    This addresses the issue described in https://github.com/aio-libs/aiohttp/pull/1453
    where Python's SimpleCookie would re-encode cookies, breaking authentication
    with certain servers.

    Args:
        cookie: A Morsel object from SimpleCookie

    Returns:
        A Morsel object with preserved coded_value

    """
    # Don't look up cookie.key in the Morsel: it maps attribute names, so a
    # cookie named ``path`` would return the Path attribute.
    mrsl_val: Morsel[str] = Morsel()
    # We use __setstate__ instead of the public set() API because it allows us to
    # bypass validation and set already validated state. This is more stable than
    # setting protected attributes directly and unlikely to change since it would
    # break pickling.
    try:
        mrsl_val.__setstate__(  # type: ignore[attr-defined]
            {
                "key": cookie.key,
                "value": cookie.value,
                "coded_value": cookie.coded_value,
            }
        )
    except CookieError:
        return cookie
    return mrsl_val


_unquote_sub = re.compile(r"\\(?:([0-3][0-7][0-7])|(.))").sub


def _unquote_replace(m: re.Match[str]) -> str:
    """
    Replace function for _unquote_sub regex substitution.

    Handles escaped characters in cookie values:
    - Octal sequences are converted to their character representation
    - Other escaped characters are unescaped by removing the backslash
    """
    if m[1]:
        return chr(int(m[1], 8))
    return m[2]


def _unquote(value: str) -> str:
    """
    Unquote a cookie value.

    Vendored from http.cookies._unquote to ensure compatibility.

    Note: The original implementation checked for None, but we've removed
    that check since all callers already ensure the value is not None.
    """
    # If there aren't any doublequotes,
    # then there can't be any special characters.  See RFC 2109.
    if len(value) < 2:
        return value
    if value[0] != '"' or value[-1] != '"':
        return value

    # We have to assume that we must decode this string.
    # Down to work.

    # Remove the "s
    value = value[1:-1]

    # Check for special sequences.  Examples:
    #    \012 --> \n
    #    \"   --> "
    #
    return _unquote_sub(_unquote_replace, value)


def parse_cookie_header(header: str) -> list[tuple[str, Morsel[str]]]:
    """
    Parse a Cookie header according to RFC 6265 Section 5.4.

    Cookie headers contain only name-value pairs separated by semicolons.
    There are no attributes in Cookie headers - even names that match
    attribute names (like 'path' or 'secure') should be treated as cookies.

    This parser uses the same regex-based approach as parse_set_cookie_headers
    to properly handle quoted values that may contain semicolons. When the
    regex fails to match a malformed cookie, it falls back to simple parsing
    to ensure subsequent cookies are not lost
    https://github.com/aio-libs/aiohttp/issues/11632

    Args:
        header: The Cookie header value to parse

    Returns:
        List of (name, Morsel) tuples for compatibility with SimpleCookie.update()
    """
    if not header:
        return []

    cookies: list[tuple[str, Morsel[str]]] = []
    morsel: Morsel[str]
    i = 0
    n = len(header)

    invalid_names = []
    while i < n:
        # Use the same pattern as parse_set_cookie_headers to find cookies
        match = _COOKIE_PATTERN.match(header, i)
        if not match:
            # Fallback for malformed cookies https://github.com/aio-libs/aiohttp/issues/11632
            # Find next semicolon to skip or attempt simple key=value parsing
            next_semi = header.find(";", i)
            eq_pos = header.find("=", i)

            # Try to extract key=value if '=' comes before ';'
            if eq_pos != -1 and (next_semi == -1 or eq_pos < next_semi):
                end_pos = next_semi if next_semi != -1 else n
                key = header[i:eq_pos].strip()
                value = header[eq_pos + 1 : end_pos].strip()

                # Validate the name (same as regex path)
                if not _COOKIE_NAME_RE.match(key):
                    invalid_names.append(key)
                else:
                    morsel = Morsel()
                    try:
                        morsel.__setstate__(  # type: ignore[attr-defined]
                            {
                                "key": key,
                                "value": _unquote(value),
                                "coded_value": value,
                            }
                        )
                    except CookieError:
                        pass
                    else:
                        cookies.append((key, morsel))

            # Move to next cookie or end
            i = next_semi + 1 if next_semi != -1 else n
            continue

        key = match.group("key")
        value = match.group("val") or ""
        i = match.end(0)

        # Validate the name
        if not key or not _COOKIE_NAME_RE.match(key):
            invalid_names.append(key)
            continue

        # Create new morsel
        morsel = Morsel()
        # Preserve the original value as coded_value (with quotes if present)
        # We use __setstate__ instead of the public set() API because it allows us to
        # bypass validation and set already validated state. This is more stable than
        # setting protected attributes directly and unlikely to change since it would
        # break pickling.
        try:
            morsel.__setstate__(  # type: ignore[attr-defined]
                {"key": key, "value": _unquote(value), "coded_value": value}
            )
        except CookieError:
            continue

        cookies.append((key, morsel))

    if invalid_names:
        internal_logger.debug(
            "Cannot load cookie. Illegal cookie names: %r", invalid_names
        )

    return cookies


def _encoded_length(text: str) -> int | None:
    """Return the UTF-8 length of text, or None if it can't be encoded."""
    if text.isascii():
        return len(text)
    try:
        return len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _attribute_value_fits(value: str) -> bool:
    """Return whether an attribute value is within the length limit."""
    length = _encoded_length(value)
    return length is not None and length <= _MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH


def parse_set_cookie_headers(headers: Sequence[str]) -> list[tuple[str, Morsel[str]]]:
    """
    Parse cookie headers using a vendored version of SimpleCookie parsing.

    This implementation is based on SimpleCookie.__parse_string to ensure
    compatibility with how SimpleCookie parses cookies, including handling
    of malformed cookies with missing semicolons.

    This function is used for both Cookie and Set-Cookie headers in order to be
    forgiving. Ideally we would have followed RFC 6265 Section 5.2 (for Cookie
    headers) and RFC 6265 Section 4.2.1 (for Set-Cookie headers), but the
    real world data makes it impossible since we need to be a bit more forgiving.

    NOTE: This implementation differs from SimpleCookie in handling unmatched quotes.
    SimpleCookie will stop parsing when it encounters a cookie value with an unmatched
    quote (e.g., 'cookie="value'), causing subsequent cookies to be silently dropped.
    This implementation handles unmatched quotes more gracefully to prevent cookie loss.
    See https://github.com/aio-libs/aiohttp/issues/7993

    At most _MAX_COOKIES_PER_RESPONSE cookies are returned; cookie-pairs over
    _MAX_COOKIE_PAIR_LENGTH octets and attribute values over
    _MAX_COOKIE_ATTRIBUTE_VALUE_LENGTH octets are ignored.
    """
    parsed_cookies: list[tuple[str, Morsel[str]]] = []

    for header in headers:
        if len(parsed_cookies) >= _MAX_COOKIES_PER_RESPONSE:
            internal_logger.debug(
                "Ignored Set-Cookie field(s) over the %d-cookie limit",
                _MAX_COOKIES_PER_RESPONSE,
            )
            break
        if not header:
            continue

        # Parse cookie string using SimpleCookie's algorithm
        i = 0
        n = len(header)
        current_morsel: Morsel[str] | None = None
        morsel_seen = False

        while 0 <= i < n:
            # Start looking for a cookie
            match = _COOKIE_PATTERN.match(header, i)
            if not match:
                # No more cookies
                break

            key, value = match.group("key"), match.group("val")
            i = match.end(0)
            lower_key = key.lower()

            if key[0] == "$":
                if not morsel_seen:
                    # We ignore attributes which pertain to the cookie
                    # mechanism as a whole, such as "$Version".
                    continue
                # Process as attribute
                if current_morsel is not None:
                    attr_lower_key = lower_key[1:]
                    if attr_lower_key in _COOKIE_KNOWN_ATTRS and _attribute_value_fits(
                        value or ""
                    ):
                        current_morsel[attr_lower_key] = value or ""
            elif lower_key in _COOKIE_KNOWN_ATTRS:
                if not morsel_seen:
                    # Invalid cookie string - attribute before cookie
                    break
                if lower_key in _COOKIE_BOOL_ATTRS:
                    # Boolean attribute with any value should be True
                    if current_morsel is not None and current_morsel.isReservedKey(key):
                        current_morsel[lower_key] = True
                elif value is None:
                    # Invalid cookie string - non-boolean attribute without value
                    break
                elif current_morsel is not None and _attribute_value_fits(value):
                    # Regular attribute with value
                    current_morsel[lower_key] = _unquote(value)
            elif value is not None:
                # This is a cookie name=value pair
                if len(parsed_cookies) >= _MAX_COOKIES_PER_RESPONSE:
                    # Stop at the per-response cookie limit.
                    internal_logger.debug(
                        "Ignored cookie(s) over the %d-cookie limit",
                        _MAX_COOKIES_PER_RESPONSE,
                    )
                    return parsed_cookies
                # Validate the name
                if key in _COOKIE_KNOWN_ATTRS or not _COOKIE_NAME_RE.match(key):
                    internal_logger.warning(
                        "Can not load cookies: Illegal cookie name %r", key
                    )
                    current_morsel = None
                elif (
                    pair_length := _encoded_length(key + value)
                ) is None or pair_length > _MAX_COOKIE_PAIR_LENGTH:
                    # Oversized or unencodable cookie-pair: ignore the cookie
                    # together with the attributes that follow it.
                    internal_logger.debug(
                        "Ignored cookie %r over the %d-octet limit",
                        key,
                        _MAX_COOKIE_PAIR_LENGTH,
                    )
                    current_morsel = None
                    morsel_seen = True
                else:
                    # Create new morsel
                    current_morsel = Morsel()
                    # Preserve the original value as coded_value (with quotes if present)
                    try:
                        current_morsel.__setstate__(  # type: ignore[attr-defined]
                            {
                                "key": key,
                                "value": _unquote(value),
                                "coded_value": value,
                            }
                        )
                    except CookieError:
                        current_morsel = None
                    else:
                        parsed_cookies.append((key, current_morsel))
                        morsel_seen = True
            else:
                # Invalid cookie string - no value for non-attribute
                break

    return parsed_cookies
