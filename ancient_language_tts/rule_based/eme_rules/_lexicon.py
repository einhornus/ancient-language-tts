"""Word lists for Early Modern English (`data/eme/phonology.txt`).

English spelling does not fix the vowel, so the spec sorts words into
lexical sets. The rule engine (`ipa.py`) reads the spelling; the lists here
say which set a word belongs to when the spelling alone cannot tell.

- `LEX` gives a whole pronunciation, for function words and irregular words.
- `STRESSED_VOWEL` gives only the vowel of the stressed syllable (its lexical
  set). The rules do the rest: consonants, unstressed vowels, syllables.
- The other sets switch one spelling rule on or off for a word.
"""

from __future__ import annotations


def _w(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Unstressed monosyllables (spec, Stress): articles, prepositions,
# conjunctions, possessives, personal pronouns.
FUNCTION = _w("""
the a an of to in on at by with for from through and or but nor that if as
my thy his her our your their its i thou thee he she it we you me him us them
ye mine thine
""")

# Auxiliaries: unstressed before another verb, stressed as the main verb.
AUX = _w("shall will do did doth have hath can may must")

# A following word of one of these kinds means the auxiliary is the main verb.
MAIN_AFTER = _w("""
the a an this that these those all no some any much many my thy his her our
your their its me him us them thee it nothing none naught more enough so too
of to in on at by with for from through such each every both what
""")

# Whole pronunciations. Function-word forms are their unstressed forms.
LEX: dict[str, str] = {
    # function words
    "the": "ðə", "a": "ə", "an": "ən", "to": "tə", "of": "ɔv", "in": "ɪn",
    "on": "ɔn", "at": "at", "by": "bəɪ", "with": "wɪθ", "for": "fɔːr",
    "from": "frɔm", "through": "θruː", "and": "and", "or": "ɔr", "but": "bʊt",
    "nor": "nɔːr", "that": "ðat", "if": "ɪf", "as": "az", "my": "məɪ",
    "thy": "ðəɪ", "his": "hɪz", "her": "hər", "our": "əʊr", "your": "joːr",
    "their": "ðeːr", "its": "ɪts", "i": "əɪ", "thou": "ðəʊ", "thee": "ðiː",
    "he": "hiː", "she": "ʃiː", "it": "ɪt", "we": "wiː", "you": "juː",
    "me": "miː", "him": "hɪm", "us": "ʊs", "them": "ðɛm", "ye": "jiː",
    "mine": "məɪn", "thine": "ðəɪn",
    "shall": "ʃal", "will": "wɪl", "do": "duː", "did": "dɪd", "doth": "dʊθ",
    "have": "hav", "hath": "haθ", "can": "kan", "may": "meː", "must": "mʊst",
    # irregular content words
    "is": "ɪz", "are": "ar", "was": "waz", "were": "wər", "been": "biːn",
    "be": "biː", "does": "dʊz", "dost": "dʊst", "doest": "ˈduː.ɪst",
    "one": "wʊn", "once": "wʊns", "two": "tuː", "who": "huː", "whom": "huːm",
    "whose": "huːz", "whole": "hoːl", "there": "ðeːr", "where": "ʍeːr",
    "than": "ðan", "though": "ðoː", "although": "ɔːlˈðoː", "o": "oː",
    "oh": "oː", "eye": "əɪ", "said": "sɛd", "says": "sɛz", "saith": "sɛθ",
    "again": "əˈgɛn", "against": "əˈgɛnst", "yes": "jɛs", "alas": "əˈlas",
    "juliet": "ˈdʒɪu.lɪ.ɛt", "england": "ˈɪŋ.glənd", "english": "ˈɪŋ.glɪʃ",
    "women": "ˈwɪ.mɪn", "woman": "ˈwʊ.mən", "could": "kʊd", "would": "wʊd",
    "should": "ʃʊd", "rough": "rʊf", "tough": "tʊf", "enough": "ɪˈnʊf",
    "cough": "kɔf", "trough": "trɔf", "plough": "pləʊ", "bough": "bəʊ",
    "dough": "doː", "thorough": "ˈθʊ.rə", "borough": "ˈbʊ.rə",
    "laugh": "laf", "laughter": "ˈlaf.tər", "draught": "draft",
    "sure": "ʃuːr", "sugar": "ˈʃʊ.gər", "salmon": "ˈsa.mən", "psalm": "sam",
    "upon": "əˈpɔn", "unto": "ˈʊn.tuː", "into": "ˈɪn.tuː",
    "within": "wɪˈðɪn", "without": "wɪˈðəʊt", "indeed": "ɪnˈdiːd",
    "always": "ˈɔːl.weːz", "also": "ˈɔːl.soː", "almost": "ˈɔːl.moːst",
    "christ": "krəɪst", "christmas": "ˈkrɪs.məs", "island": "ˈəɪ.lənd",
    "isle": "əɪl", "height": "həɪt", "only": "ˈoːn.lɪ", "early": "ˈər.lɪ",
    "nothing": "ˈnʊ.θɪn", "something": "ˈsʊm.θɪn", "anything": "ˈɛ.nɪ.θɪn",
    "everything": "ˈɛv.rɪ.θɪn", "every": "ˈɛv.rɪ", "evening": "ˈiːv.nɪn",
    "people": "ˈpiː.pəl", "water": "ˈwa.tər", "sword": "soːrd",
    "answer": "ˈan.sər", "eleven": "ɪˈlɛ.vən", "foreign": "ˈfɔ.rɪn",
    "sovereign": "ˈsɔv.rɪn", "choir": "kwəɪr", "mischief": "ˈmɪs.tʃɪf",
    "tongue": "tʊŋ", "leopard": "ˈlɛ.pərd", "jeopardy": "ˈdʒɛ.pər.dɪ",
    "yeoman": "ˈjoː.mən", "beyond": "bɪˈjɔnd", "perhaps": "pərˈhaps",
    "often": "ˈɔf.tən", "pretty": "ˈprɪ.tɪ", "busy": "ˈbɪ.zɪ",
    "business": "ˈbɪz.nɪs", "heaven": "ˈhɛ.vən", "even": "ˈiː.vən",
    "ever": "ˈɛ.vər", "spirit": "ˈspɪ.rɪt", "power": "ˈpəʊ.ər",
    "knowledge": "ˈknɔ.lɪdʒ", "colonel": "ˈkər.nəl", "lieu": "lɪu",
    "friend": "frɛnd", "shoe": "ʃuː", "canoe": "kəˈnuː", "won't": "woːnt",
    "don't": "doːnt", "can't": "kant", "aught": "ɔːt", "naught": "nɔːt",
    # apostrophe forms (spec: the lost syllable is not pronounced)
    "heav'n": "hɛvn", "o'er": "oːr", "e'er": "eːr", "ne'er": "neːr",
    "e'en": "iːn", "'tis": "tɪz", "'twas": "twaz", "th'": "ð",
    "ta'en": "teːn", "giv'n": "gɪvn", "ev'n": "iːvn", "ev'ry": "ˈɛv.rɪ",
    "i'": "ɪ", "o'": "ɔ",
    # Shakespeare's stress (spec, Stress)
    "aspect": "aˈspɛkt", "revenue": "rɪˈvɛ.nɪu", "persever": "pərˈsɛ.vər",
    "obdurate": "ɔbˈdɪu.rət", "canonized": "kaˈnɔ.nəɪzd",
    "canonize": "kaˈnɔ.nəɪz", "advertised": "adˈvər.təɪzd",
    "advertise": "adˈvər.təɪz", "commendable": "ˈkɔ.mən.də.bəl",
    "triumphing": "trəɪˈʊm.fɪn", "sepulchre": "sɪˈpʊl.kər",
    "contrary": "kənˈtreː.rɪ", "record": "rɪˈkɔːrd", "exile": "ɛkˈsəɪl",
    "complete": "kəmˈpliːt", "extreme": "ɛkˈstriːm", "employ": "əmˈplɔɪ",
    "obey": "əˈbeː", "wherefore": "ˈʍeːr.foːr", "therefore": "ˈðeːr.foːr",
}

# Before a stressed noun (spec): complete [ˈkɔm.pliːt], extreme [ˈɛk.striːm].
ATTRIBUTIVE = {"complete": "ˈkɔm.pliːt", "extreme": "ˈɛk.striːm"}

# In verse, one syllable when the meter needs it (spec).
VERSE_SHORT = {
    "heaven": "ˈhɛvn", "even": "ˈiːn", "ever": "ˈeːr", "spirit": "ˈspɪrt",
    "power": "ˈpəʊr",
}

# The vowel of the stressed syllable, by lexical set.
_SETS = {
    "ɛ": """head dead bread breath death deaf dread spread thread tread instead
        heavy ready steady weather feather leather heather measure pleasure
        treasure meadow pleasant peasant health wealth stealth sweat threat
        breakfast jealous zealous realm endeavour meant dealt dreamt leapt
        cleanse breast weapon already any many very never sever clever seven
        devil level merit perish cherish relish credit edit tenant petal metal
        medal second bury present presence precious peril treachery heifer
        breadth""",
    "ɪ": """wind give live forgive river shiver liver quiver given driven riven
        visit limit city pity finish lily timid vivid figure sinew linen sieve
        minute widow prison risen civil consider continue build built""",
    "a": """father rather gather lather shadow image damage palace manage magic
        camel panel planet cabin salad habit travel gravel rapid valid panic
        dragon wagon flagon talent valour chapel vanish banish ravish aunt
        family""",
    "ɔ": """gone shone body copy honour honor honest profit proper robin comic
        model modern prophet solid olive promise closet forest florid orange
        coral moral novel column colony""",
    "ʊ": """love dove glove above shove come become some done none son ton won
        front month monk money honey other mother brother smother cover covet
        among wonder worry oven govern comfort company compass onion colour
        color dozen sponge monday london blood flood good foot book look took
        cook hook wood stood hood wool brook crook shook soot understood couple
        double trouble country cousin touch young southern nourish flourish
        courage woman wolf bosom hover constable conjure comely covenant""",
    "oː": """both most post ghost host soul shoulder mould poultry boulder comb
        sew shew four pour course court source mourn fourth door floor sworn
        worn torn shorn borne forth port sport fort force ford porch pork
        support report afford portion know known grow grown own owner show
        shown slow low blow blown snow flow throw thrown crow row mow sow tow
        glow owe bowl growth elbow below bestow flown sown mown""",
    "uː": """move prove lose whose tomb womb youth soup group wound route rouge
        approve remove""",
    "ɔː": "broad abroad",
    "ə": """heard earth learn earn search pearl earnest yearn dearth rehearse
        journey journal courtesy courteous adjourn sojourn""",
    "aː": "heart hearth",
    "eː": """bear wear pear swear forbear forswear ancient chamber angel""",
    "əɪ": "sleight pint climb",
}
STRESSED_VOWEL: dict[str, str] = {}
for _v, _text in _SETS.items():
    for _word in _text.split():
        STRESSED_VOWEL[_word] = "a" if _v == "aː" else _v

# Prefixes that are not prefixes in these words (stress the first syllable).
NO_PREFIX = _w("""
devil delicate decent recent region rebel relic remedy reverence reverend
desert deluge beckon benefit decade delegate demon regal recompense renegade
reprobate requiem residue resonant revel relative regiment reference
preference president prelate precept precedent premise preface precious
presence present pretty distance district discord discipline dismal distant
distaff mischief miser misery mistress mister misty under uncle union unit
unity universe unicorn combat comfort common commons comely company compass
comment commerce compound concord conduct conquer conquest conscience
consequence constant constable contract convent conjure confident conference
conflict congress consul concert comedy comet complex comrade consonant
contemplate compliment admiral adamant advocate advent adverb adder object
obstacle obvious obstinate obloquy oblong obsolete perfect person perfume
perjury perilous permanent pertinent persecute promise profit product
province prophet proper property prologue prodigal progress prospect
provender protestant prophecy provost proverb prowess process prosper
prostrate excellent exercise exquisite exit
""")

# Words with the a- prefix [ə].
A_WORDS = _w("""
about above alone among away awake awhile arise asleep afraid aloud ago
along amaze abide abroad across afar afoot afore aground ahead alike alive
aloft aloof amid amidst amiss anew apart aright around arose ashamed ashore
aside astray athwart awry aware abode abound awoke
""")

WITH_WORDS = _w("withal withdraw withhold withstand")
FOR_WORDS = _w("forget forgive forbid forsake forswear forbear forgot forgotten forgiven forbade forlorn")

# g before e, i, y that stays [g].
HARD_G = _w("""
get give girl gift gild gird girdle begin began begun beget forget forgive
together tiger eager finger linger anger hunger longer longest stronger
strongest younger youngest geese gear giddy gilt target gibbous giggle
forgotten getting gotten
""")

# Verbs whose -er keeps [ŋ]: singer [ˈsɪŋ.ər].
SING_VERBS = _w("sing bring ring hang wing sting swing fling cling spring wring string")

# Intervocalic s that stays [s].
VOICELESS_S = _w("""
lease cease case base chase dose close goose loose mouse house grease crease
geese purpose promise precise concise paradise increase decrease morose
verbose premise basis crisis obese
""")

# ch = [k].
CH_K = _w("""
chorus christ christian christen character choir chaos chord echo school
scholar stomach monarch ache anchor architect orchestra chemist michael chasm
chimera chronicle
""")

WH_H = _w("who whom whose whole wholly whoever whore wholesome")

# gh = [f], as in the modern word (spec).
GH_F = _w("laugh enough rough cough tough trough draught laughter")

# Initial h silent in French loans (spec). Matched as prefixes of the word.
FRENCH_H = ("hour", "honest", "honour", "honor", "heir", "herb", "humbl",
            "humble", "humour", "humor", "hospital", "host")

# th = [ð] at the start of these words (spec).
TH_VOICED = _w("""
the this that these those then than there they them their thee thou thy thine
though thus theirs themselves thyself therefore thereby therein thereof
thereon thereto thence thither
""")

# Medial th that is [θ] although between vowels.
THETA_MEDIAL = _w("""
author method catholic ether athens ethiop sympathy apathy gothic lethal
arthur mathematics ethic matthew timothy bethlehem
""")

# -ic words without stress on the syllable before -ic.
IC_INITIAL = _w("catholic politic lunatic heretic arithmetic rhetoric arabic choleric bishopric")

# ea as two vowels.
EA_HIATUS = ("real", "idea", "create", "creati", "theat", "area")
# ui as two vowels.
UI_HIATUS = ("ruin", "flui", "drui", "intui")

# -ed that is part of the word, not a suffix.
ED_ROOT = _w("""
hundred kindred sacred naked wicked wretched crooked rugged ragged hatred
shred creed deed need seed speed steed weed bleed breed feed greed heed
indeed proceed succeed exceed
""")

# Original spellings (spec: doe is do, beene is been).
ORIGINAL = {
    "doe": "do", "goe": "go", "soe": "so", "noe": "no", "beene": "been",
    "bin": "been", "mee": "me", "hee": "he", "shee": "she", "wee": "we",
    "yee": "ye", "onely": "only", "heere": "here", "deere": "dear",
    "cleere": "clear", "musick": "music", "publick": "public",
    "magick": "magic", "doo": "do", "shew": "show",
}
