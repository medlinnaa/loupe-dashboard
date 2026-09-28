"""Loupe: Bina.az bargain finder dashboard.

Reads predictions.json (produced by model/train.py) and lets a visitor
check a listing, browse the best deals and see how the model behaves.
"""
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).parent / "predictions.json"

# model/train.py holds out 20% of the rows with train_test_split(random_state=42).
# The split depends only on the row count, so we can tell which listings the
# model never saw. We only trust this while the file has the size train.py used.
SPLIT_N, SPLIT_SEED, SPLIT_FRAC = 1517, 42, 0.2

INK, MUTED = "#14202B", "#5B6B78"
COLOR = {"vcheap": "#0B7A75", "below": "#6DBBB4", "fair": "#8794A0", "above": "#B4463C"}
# Result-card shades per verdict: card background, pill background, pill text
CARD_BG = {"vcheap": "#EAF6F4", "below": "#F2F9F8", "fair": "#FFFFFF", "above": "#FCF3F2"}
PILL = {"vcheap": ("#0B7A75", "#FFFFFF"), "below": ("#CFEAE7", "#0B5D63"),
        "fair": ("#E4E9ED", "#33424F"), "above": ("#F3D6D2", "#8E2F27")}
# Dots on the dark hero band, lighter than the chart colors so they stay visible
HERO_DOT = {"total": "#FFFFFF", "vcheap": "#5FD4C8", "below": "#BFE3DF", "above": "#E58C82"}
TEAL_SCALE = ["#D6EEEB", "#6DBBB4", "#0B7A75", "#08343B"]
TEXT_COLOR = {"vcheap": "#0B7A75", "below": "#0B7A75", "fair": INK, "above": "#B4463C"}
VERDICT_ORDER = ["vcheap", "below", "fair", "above"]

AREA_NAMES = {
    "hezi-aslanov": "Həzi Aslanov", "28-may": "28 May", "qara-qarayev": "Qara Qarayev",
    "elmler-akademiyasi": "Elmlər Akademiyası", "neriman-nerimanov": "Nəriman Nərimanov",
    "xetai": "Xətai", "insaatcilar": "İnşaatçılar", "abseron": "Abşeron",
    "yasamal": "Yasamal", "suraxani": "Suraxanı", "ehmedli": "Əhmədli",
    "sabuncu": "Sabunçu", "memar-ecemi": "Memar Əcəmi", "nizami-metrosu": "Nizami m.",
    "bineqedi": "Binəqədi", "sah-ismayil-xetai": "Şah İsmayıl Xətai", "nesimi": "Nəsimi",
    "neftciler": "Neftçilər", "nerimanov": "Nərimanov", "azadliq-prospekti": "Azadlıq prospekti",
    "sebail": "Səbail", "xalqlar-dostlugu": "Xalqlar Dostluğu", "genclik": "Gənclik",
    "20-yanvar": "20 Yanvar", "8-noyabr": "8 Noyabr", "koroglu": "Koroğlu", "sahil": "Sahil",
    "nesimi-metrosu": "Nəsimi m.", "iceri-seher-metrosu": "İçəri Şəhər m.", "nizami": "Nizami",
    "dernegul": "Dərnəgül", "qaradag": "Qaradağ", "bakmil": "Bakmil",
    "avtovagzal": "Avtovağzal", "xezer": "Xəzər",
}

STR = {
    "AZ": {
        "tagline": "Bakıda bazar qiymətindən ucuz satılan mənzilləri tapın.",
        "stat_total": "elan təhlil olunub",
        "stat_vc": "çox ucuz (15%+)",
        "stat_bm": "ucuz (10–15%)",
        "stat_ab": "bahalı (10%+)",
        "badge_vcheap": "Çox ucuz", "badge_below": "Ucuz",
        "badge_fair": "Bazar səviyyəsində", "badge_above": "Bahalı",
        "lookup": "Bina.az elanının linki və ya nömrəsi",
        "btn_random": "Təsadüfi elan",
        "btn_best": "Ən sərfəli elan",
        "bad_input": "Düzgün Bina.az linki və ya elan nömrəsi daxil edin. Məsələn: https://bina.az/items/6364416",
        "not_found": "Bu elan bazada tapılmadı. Bazada yalnız {n} elan var, nümunə üçün yuxarıdakı düymələrdən birini seçin.",
        "empty": "Bir elanın linkini yapışdırın və ya nümunə seçin.",
        "apt": "{rooms} otaqlı mənzil, {size} m²",
        "listed": "Elan qiyməti",
        "model": "Model proqnozu",
        "per_sqm": "Elan: {a} AZN/m². Model: {p} AZN/m².",
        "v_vcheap": "Bazar qiymətindən {p} ucuzdur, çox sərfəli təklifdir ({d} AZN fərq).",
        "v_below": "Bazar qiymətindən {p} ucuzdur ({d} AZN fərq).",
        "v_fair_lo": "Qiymət bazar səviyyəsinə yaxındır, {p} ucuzdur.",
        "v_fair_hi": "Qiymət bazar səviyyəsinə yaxındır, {p} bahadır.",
        "v_above": "Bazar qiymətindən {p} bahadır ({d} AZN fərq).",
        "seen_no": "Model bu elanı öyrənmə zamanı görməyib, ona görə proqnoz müstəqil yoxlama kimi etibarlıdır.",
        "seen_yes": "Model bu elanı öyrənmə zamanı görüb, ona görə proqnoz olduğundan dəqiq görünə bilər.",
        "open": "Bina.az-da aç",
        "filters": "Filtrlər",
        "f_sub": "Aşağıdakı cədvəl və qrafiklərə tətbiq olunur.",
        "f_count": "elan filtrlərə uyğundur",
        "f_reset": "Filtrləri sıfırla",
        "f_area": ":material/location_on: Ərazi",
        "f_rooms": ":material/bed: Otaq sayı",
        "f_price": ":material/payments: Qiymət aralığı (AZN)",
        "f_held": ":material/visibility_off: Yalnız modelin görmədiyi elanlar ({n})",
        "f_held_help": "Model bu elanlar üzərində öyrədilməyib, ona görə onlar üçün proqnoz daha etibarlıdır.",
        "tab_deals": "Ən sərfəli elanlar",
        "tab_dist": "Paylanma",
        "tab_areas": "Ərazilər",
        "min_disc": "Ən azı bu qədər ucuz (%)",
        "shown": "{k} elan göstərilir, ən sərfəlisindən başlayaraq.",
        "no_rows": "Bu filtrlərə uyğun elan yoxdur. Filtrləri genişləndirin.",
        "c_area": "Ərazi", "c_rooms": "Otaq", "c_size": "Sahə (m²)", "c_price": "Qiymət (AZN)",
        "c_model": "Model (AZN)", "c_score": "Ucuzluq (%)", "c_unseen": "Modelin görmədiyi",
        "c_link": "Elan", "link_text": "Aç",
        "cat_vcheap": "Çox ucuz (15%-dən çox)", "cat_below": "Ucuz (10–15%)",
        "cat_fair": "Bazar səviyyəsində", "cat_above": "Bazardan bahalı (10%-dən çox)",
        "dist_title": "Elanlar modelin qiymətinə görə necə paylanıb",
        "x_score": "Ucuzluq (%). Müsbət dəyər modelin qiymətindən ucuz deməkdir",
        "y_count": "Elan sayı",
        "clipped": "{n} elanın ucuzluğu ±60% hüdudundan kənardadır və qrafikdə həddə göstərilir.",
        "scatter_title": "Elan qiyməti ilə model proqnozu",
        "x_price": "Elan qiyməti (AZN)", "y_pred": "Model proqnozu (AZN)",
        "equal": "Bərabər qiymət",
        "acc": "Modelin görmədiyi {n} elanda orta nisbi xəta {mape}, median xəta {med}. Bu xəta alarm hədləri (10% və 15%) ilə eyni tərtibdədir, ona görə nəticəni “yoxlamağa dəyər” siqnalı kimi oxuyun.",
        "area_ppsm": "Ərazi üzrə median qiymət (AZN/m²)",
        "area_cheap": "Ərazi üzrə ucuz elanların payı (ucuzluq 10% və daha çox)",
        "x_share": "Ərazidəki elanların payı (%)",
        "area_note": "Ən azı {k} elanı olan ərazilər göstərilir. Mötərizədə: ucuz elan / ərazidəki bütün elanlar.",
        "area_none": "Ərazi qrafiki üçün hər ərazidə ən azı {k} elan lazımdır. Filtrləri genişləndirin.",
        "footer": "Holberton School kapstoun layihəsi. Məlumat Bina.az-dan yığılmış elanların statik nüsxəsidir, elan artıq silinmiş və ya qiyməti dəyişmiş ola bilər. Ucuzluq = (model proqnozu − elan qiyməti) / model proqnozu × 100.",
        "unit_sqm": "AZN/m²",
    },
    "EN": {
        "tagline": "Find apartments in Baku that are listed below their market price.",
        "stat_total": "listings analysed",
        "stat_vc": "very cheap (15%+)",
        "stat_bm": "cheap (10–15%)",
        "stat_ab": "above market (10%+)",
        "badge_vcheap": "Very cheap", "badge_below": "Cheap",
        "badge_fair": "Near market", "badge_above": "Above market",
        "lookup": "Bina.az listing link or number",
        "btn_random": "Random listing",
        "btn_best": "Best deal",
        "bad_input": "Enter a valid Bina.az link or listing number, for example https://bina.az/items/6364416",
        "not_found": "This listing is not in the dataset. It only holds {n} listings, so pick one with the buttons above.",
        "empty": "Paste a listing link or pick an example.",
        "apt": "{rooms}-room apartment, {size} m²",
        "listed": "Listed price",
        "model": "Model price",
        "per_sqm": "Listed: {a} AZN/m². Model: {p} AZN/m².",
        "v_vcheap": "{p} below the market price, a very good deal ({d} AZN difference).",
        "v_below": "{p} below the market price ({d} AZN difference).",
        "v_fair_lo": "Close to the market price, {p} cheaper.",
        "v_fair_hi": "Close to the market price, {p} more expensive.",
        "v_above": "{p} above the market price ({d} AZN difference).",
        "seen_no": "The model did not see this listing during training, so the prediction is an independent check.",
        "seen_yes": "The model saw this listing during training, so its prediction may look more accurate than it really is.",
        "open": "Open on Bina.az",
        "filters": "Filters",
        "f_sub": "Applies to the table and charts below.",
        "f_count": "listings match your filters",
        "f_reset": "Reset filters",
        "f_area": ":material/location_on: Area",
        "f_rooms": ":material/bed: Rooms",
        "f_price": ":material/payments: Price range (AZN)",
        "f_held": ":material/visibility_off: Only listings the model has not seen ({n})",
        "f_held_help": "The model was not trained on these listings, so its predictions for them are more trustworthy.",
        "tab_deals": "Best deals",
        "tab_dist": "Distribution",
        "tab_areas": "Areas",
        "min_disc": "At least this much cheaper (%)",
        "shown": "Showing {k} listings, best deal first.",
        "no_rows": "No listing matches these filters. Widen them.",
        "c_area": "Area", "c_rooms": "Rooms", "c_size": "Size (m²)", "c_price": "Price (AZN)",
        "c_model": "Model (AZN)", "c_score": "Discount (%)", "c_unseen": "Unseen by model",
        "c_link": "Listing", "link_text": "Open",
        "cat_vcheap": "Very cheap (over 15%)", "cat_below": "Cheap (10–15%)",
        "cat_fair": "Near market", "cat_above": "Above market (over 10%)",
        "dist_title": "How listings are spread around the model's price",
        "x_score": "Discount (%). Positive means cheaper than the model's price",
        "y_count": "Listings",
        "clipped": "{n} listings have a discount beyond ±60% and are drawn at the edge of the chart.",
        "scatter_title": "Listed price vs model price",
        "x_price": "Listed price (AZN)", "y_pred": "Model price (AZN)",
        "equal": "Equal price",
        "acc": "On the {n} listings the model has not seen, the mean relative error is {mape} and the median is {med}. That is the same size as the alert thresholds (10% and 15%), so read a result as “worth a closer look”.",
        "area_ppsm": "Median price by area (AZN/m²)",
        "area_cheap": "Share of cheap listings by area (discount of 10% or more)",
        "x_share": "Share of the area's listings (%)",
        "area_note": "Showing areas with at least {k} listings. In brackets: cheap listings / all listings in the area.",
        "area_none": "The area charts need at least {k} listings per area. Widen the filters.",
        "footer": "Holberton School capstone project. The data is a static copy of Bina.az listings, so a listing may since have been removed or repriced. Discount = (model price − listed price) / model price × 100.",
        "unit_sqm": "AZN/m²",
    },
}

st.set_page_config(page_title="Loupe", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
[data-testid="stMainBlockContainer"]{max-width:1080px;padding-top:1.6rem}
.lq-hero,.lq-hero *,.lq-result,.lq-result *{font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.lq-hero{position:relative;overflow:hidden;border-radius:16px;padding:2.7rem 2.8rem 2.3rem;margin:0 0 1.8rem;color:#FFFFFF;background:radial-gradient(110% 150% at 100% 0%,rgba(11,122,117,.85) 0%,rgba(11,122,117,0) 58%),linear-gradient(135deg,#06262C 0%,#0A4A52 60%,#0B5D63 100%)}
.lq-hero:after{content:"";position:absolute;right:-60px;bottom:-95px;width:270px;height:270px;border-radius:50%;border:36px solid rgba(255,255,255,.07);pointer-events:none}
.lq-hero p{margin:0}
.lq-hero p.lq-hero-title{color:#FFFFFF;font-size:clamp(3.6rem,10vw,6rem);font-weight:700;letter-spacing:-0.04em;line-height:.92}
.lq-hero p.lq-hero-tag{color:#E3F3F1;font-size:1.25rem;font-weight:400;line-height:1.35;max-width:34rem;margin-top:1rem}
.lq-stats{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:.7rem;margin-top:1.8rem}
.lq-stat{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.17);border-radius:12px;padding:.7rem 1.05rem .75rem;min-width:8.6rem}
.lq-stat b{display:block;font-size:1.7rem;font-weight:600;letter-spacing:-.01em;line-height:1.1;color:#FFFFFF}
.lq-stat span{display:flex;align-items:center;gap:.45rem;font-size:.85rem;color:#E3F3F1;margin-top:.3rem}
.lq-stat i{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;flex:none}
.lq-result{border-radius:12px;padding:1.5rem 1.7rem 1.4rem;margin:1.1rem 0 .6rem;color:#14202B;border:1px solid #D5DCE2;border-top-width:5px}
.lq-result p{color:#14202B;margin:0}
.lq-head{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.lq-result p.lq-r-title{font-size:1.35rem;font-weight:600;line-height:1.25}
.lq-result p.lq-r-sub{color:#5B6B78;margin-top:.15rem}
.lq-pill{flex:none;border-radius:999px;padding:.3rem .85rem;font-size:.85rem;font-weight:600;white-space:nowrap}
.lq-result p.lq-verdict{font-size:1.05rem;font-weight:500;margin-top:1.1rem;line-height:1.4}
.lq-strip{position:relative;height:108px;margin:.9rem .9rem .2rem}
.lq-track{position:absolute;left:0;right:0;top:46px;height:6px;background:rgba(20,32,43,.12);border-radius:3px}
.lq-gap{position:absolute;top:0;height:6px;border-radius:3px}
.lq-tick{position:absolute;top:-9px;width:2px;height:24px;background:#14202B;transform:translateX(-1px)}
.lq-dot{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;transform:translateX(-8px);border:3px solid #FFFFFF;box-sizing:border-box;box-shadow:0 0 0 1.5px #14202B}
.lq-lab{position:absolute;transform:translateX(-50%);text-align:center;white-space:nowrap;font-size:.85rem;line-height:1.3;color:#5B6B78}
.lq-lab b{display:block;font-weight:600;font-size:.95rem;color:#14202B}
.lq-result p.lq-meta{font-size:.9rem;color:#5B6B78;margin-top:.4rem}
.lq-result a.lq-link{display:inline-block;margin-top:.9rem;color:#0B5D63;font-weight:600;text-decoration:underline;text-underline-offset:3px}
.stButton>button{border-radius:999px;border:1px solid #0B7A75;color:#0B5D63;background:#FFFFFF;font-weight:500}
.stButton>button:hover{background:#E4F2F0;border-color:#0B5D63;color:#08343B}
button[data-baseweb="tab"] p{font-size:1.02rem;font-weight:500}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#E8F3F1 0%,#D2E7E3 100%)}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:transparent}
.lq-side-head,.lq-side-head *,.lq-count,.lq-count *{font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif}
.lq-side-head{margin:.2rem 0 .3rem}
.lq-side-head p{margin:0}
.lq-side-head p.lq-side-title{color:#08343B;font-size:1.4rem;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.lq-side-head p.lq-side-sub{color:#2F4F55;font-size:.88rem;line-height:1.35;margin-top:.25rem}
.lq-count{border-radius:14px;padding:1rem 1.15rem 1.1rem;color:#FFFFFF;background:linear-gradient(135deg,#06262C 0%,#0A4A52 65%,#0B5D63 100%)}
.lq-count p{margin:0}
.lq-count p.lq-count-num{color:#FFFFFF;font-size:2.2rem;font-weight:700;letter-spacing:-.02em;line-height:1}
.lq-count p.lq-count-num span{color:#BFE3DF;font-size:1rem;font-weight:500;letter-spacing:0}
.lq-count p.lq-count-lab{color:#E3F3F1;font-size:.88rem;margin-top:.4rem}
.lq-count-bar{height:6px;border-radius:3px;background:rgba(255,255,255,.18);margin-top:.85rem;overflow:hidden}
.lq-count-bar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,#5FD4C8,#BFE3DF)}
.st-key-reset_box{margin-top:.8rem}
.st-key-card_where,.st-key-card_price,.st-key-card_model{background:#FFFFFF;border:1px solid #BCD8D3;border-radius:12px;padding:.95rem 1rem .6rem}
.st-key-card_where [data-baseweb="select"]>div{background:#F1F7F6;border-radius:10px}
</style>
""",
    unsafe_allow_html=True,
)

# Language picker first: every other label depends on it.
L = st.sidebar.radio("Dil / Language", ["AZ", "EN"], horizontal=True, key="lang")


def t(key, **kw):
    return STR[L][key].format(**kw)


def fnum(x, d=0):
    """Format a number the way the chosen language writes it."""
    s = f"{x:,.{d}f}"
    if L == "AZ":
        s = s.replace(",", "\u00a0").replace(".", ",")
    return s


def fpct(x, d=1):
    return f"{fnum(x, d)}%"


def verdict_of(score):
    if score >= 15:
        return "vcheap"
    if score >= 10:
        return "below"
    if score > -10:
        return "fair"
    return "above"


@st.cache_data(show_spinner=False)
def load_data():
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(list(raw.values()))
    n = len(df)
    split_ok = n == SPLIT_N
    held = np.zeros(n, dtype=bool)
    if split_ok:
        test_idx = np.random.RandomState(SPLIT_SEED).permutation(n)[: math.ceil(SPLIT_FRAC * n)]
        held[test_idx] = True
    df["held_out"] = held
    df["area_key"] = df["district"]
    df["area_name"] = df["district"].map(
        lambda s: "Naməlum" if s in ("nan", "", None) else AREA_NAMES.get(s, str(s).replace("-", " ").title())
    )
    df["ppsm"] = df["actual_price"] / df["area"]
    df["pred_ppsm"] = df["predicted_price"] / df["area"]
    df["verdict"] = df["bargain_score"].map(verdict_of)
    df["ape"] = (df["predicted_price"] - df["actual_price"]).abs() / df["actual_price"] * 100
    return df, split_ok


def parse_id(text):
    m = re.search(r"items/(\d+)", text)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d{5,9}", text):
        return int(text)
    return None


def price_strip(actual, pred, color):
    """One line showing listed price (dot) against model price (tick)."""
    top = max(actual, pred) * 1.18
    a, p = actual / top * 100, pred / top * 100
    lo, hi = min(a, p), max(a, p)

    def clamp(x):
        return min(max(x, 14), 86)

    return (
        '<div class="lq-strip">'
        f'<div class="lq-lab" style="left:{clamp(p):.1f}%;top:0">{t("model")}<b>{fnum(pred)} AZN</b></div>'
        '<div class="lq-track">'
        f'<div class="lq-gap" style="left:{lo:.1f}%;width:{hi - lo:.1f}%;background:{color}"></div>'
        f'<div class="lq-tick" style="left:{p:.1f}%"></div>'
        f'<div class="lq-dot" style="left:{a:.1f}%;background:{color}"></div>'
        "</div>"
        f'<div class="lq-lab" style="left:{clamp(a):.1f}%;top:70px">{t("listed")}<b>{fnum(actual)} AZN</b></div>'
        "</div>"
    )


def render_result(r, split_ok):
    v = r["verdict"]
    score = float(r["bargain_score"])
    diff = abs(float(r["predicted_price"]) - float(r["actual_price"]))
    if v == "fair":
        text = t("v_fair_lo" if score >= 0 else "v_fair_hi", p=fpct(abs(score)))
    else:
        text = t(f"v_{v}", p=fpct(abs(score)), d=fnum(diff))
    seen = ""
    if split_ok:
        seen = f'<p class="lq-meta">{t("seen_no" if r["held_out"] else "seen_yes")}</p>'
    pill_bg, pill_fg = PILL[v]
    title = t("apt", rooms=int(r["rooms"]), size=fnum(r["area"], 0 if float(r["area"]).is_integer() else 1))
    html = (
        f'<div class="lq-result" style="background:{CARD_BG[v]};border-top-color:{COLOR[v]}">'
        '<div class="lq-head"><div>'
        f'<p class="lq-r-title">{title}</p>'
        f'<p class="lq-r-sub">{r["area_name"]}</p>'
        '</div>'
        f'<span class="lq-pill" style="background:{pill_bg};color:{pill_fg}">{t(f"badge_{v}")}</span>'
        '</div>'
        f'<p class="lq-verdict" style="color:{TEXT_COLOR[v]}">{text}</p>'
        + price_strip(float(r["actual_price"]), float(r["predicted_price"]), COLOR[v])
        + f'<p class="lq-meta">{t("per_sqm", a=fnum(r["ppsm"]), p=fnum(r["pred_ppsm"]))}</p>'
        + seen
        + f'<a class="lq-link" href="{r["url"]}" target="_blank" rel="noopener noreferrer">{t("open")}</a>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


PRICE_TICKS_COARSE = [20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
PRICE_TICKS_FINE = [20_000, 30_000, 40_000, 50_000, 70_000, 100_000, 150_000, 200_000, 300_000,
                    400_000, 500_000, 700_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000]


def price_ticks(lo, hi):
    """Round price ticks (50k, 100k, 1M...) for a log axis, instead of Plotly's bare digits."""
    ticks = [v for v in PRICE_TICKS_COARSE if lo * 0.85 <= v <= hi * 1.15]
    if len(ticks) < 4:
        ticks = [v for v in PRICE_TICKS_FINE if lo * 0.85 <= v <= hi * 1.15]
    labels = [f"{v / 1e6:g}M" if v >= 1_000_000 else f"{v / 1e3:g}k" for v in ticks]
    return ticks, labels


def chart_layout(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=8, b=0),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------- data + sidebar
df, split_ok = load_data()
n_held = int(df["held_out"].sum())

# Price slider works in thousands of AZN so its ends read "45k" and "2500k".
P_LO_K = (int(df["actual_price"].min()) // 5000) * 5
P_HI_K = -(-int(df["actual_price"].max()) // 5000) * 5
# Reset works by bumping a version number that is part of every filter widget's key,
# so the widgets are recreated at their defaults (no Session State writes needed).
flt_ver = st.session_state.setdefault("flt_ver", 0)


def reset_filters():
    st.session_state["flt_ver"] += 1


st.sidebar.markdown(
    f'<div class="lq-side-head"><p class="lq-side-title">{t("filters")}</p>'
    f'<p class="lq-side-sub">{t("f_sub")}</p></div>',
    unsafe_allow_html=True,
)
count_slot = st.sidebar.container()  # filled in once the filters are known
with st.sidebar.container(key="card_where"):
    sel_areas = st.multiselect(t("f_area"), sorted(df["area_name"].unique()), key=f"f_area_{flt_ver}")
    sel_rooms = st.multiselect(t("f_rooms"), sorted(int(x) for x in df["rooms"].unique()), key=f"f_rooms_{flt_ver}")
with st.sidebar.container(key="card_price"):
    price_k = st.slider(t("f_price"), P_LO_K, P_HI_K, (P_LO_K, P_HI_K), step=5, format="%dk", key=f"f_price_{flt_ver}")
price_rng = (price_k[0] * 1000, price_k[1] * 1000)
only_unseen = False
if split_ok:
    with st.sidebar.container(key="card_model"):
        only_unseen = st.toggle(t("f_held", n=n_held), help=t("f_held_help"), key=f"f_held_{flt_ver}")

fdf = df[df["actual_price"].between(*price_rng)]
if sel_areas:
    fdf = fdf[fdf["area_name"].isin(sel_areas)]
if sel_rooms:
    fdf = fdf[fdf["rooms"].isin(sel_rooms)]
if only_unseen:
    fdf = fdf[fdf["held_out"]]

filters_active = bool(sel_areas or sel_rooms or only_unseen or tuple(price_k) != (P_LO_K, P_HI_K))
with count_slot:
    st.markdown(
        '<div class="lq-count">'
        f'<p class="lq-count-num">{fnum(len(fdf))}<span> / {fnum(len(df))}</span></p>'
        f'<p class="lq-count-lab">{t("f_count")}</p>'
        f'<div class="lq-count-bar"><i style="width:{len(fdf) / len(df) * 100:.1f}%"></i></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="reset_box"):
        st.button(t("f_reset"), on_click=reset_filters, disabled=not filters_active, key="f_reset", width="stretch")

# ---------------------------------------------------------------- hero
n_vc = int((df["bargain_score"] >= 15).sum())
n_bm = int(((df["bargain_score"] >= 10) & (df["bargain_score"] < 15)).sum())
n_ab = int((df["bargain_score"] <= -10).sum())


def stat(number, key, dot):
    return f'<div class="lq-stat"><b>{fnum(number)}</b><span><i style="background:{HERO_DOT[dot]}"></i>{t(key)}</span></div>'


st.markdown(
    '<div class="lq-hero">'
    '<p class="lq-hero-title">Loupe</p>'
    f'<p class="lq-hero-tag">{t("tagline")}</p>'
    '<div class="lq-stats">'
    + stat(len(df), "stat_total", "total")
    + stat(n_vc, "stat_vc", "vcheap")
    + stat(n_bm, "stat_bm", "below")
    + stat(n_ab, "stat_ab", "above")
    + "</div></div>",
    unsafe_allow_html=True,
)

pool = df[df["held_out"]] if split_ok else df
default_url = pool.loc[pool["bargain_score"].idxmax(), "url"]
best_url = df.loc[df["bargain_score"].idxmax(), "url"]
all_urls = df["url"].tolist()


def pick_random():
    st.session_state["q"] = random.choice(all_urls)


def pick_best():
    st.session_state["q"] = best_url


st.session_state.setdefault("q", default_url)
st.text_input(t("lookup"), key="q", placeholder="https://bina.az/items/6364416")
b1, b2, _ = st.columns([1.2, 1.2, 4])
b1.button(t("btn_random"), on_click=pick_random, width="stretch")
b2.button(t("btn_best"), on_click=pick_best, width="stretch")

query = st.session_state["q"].strip()
if not query:
    st.caption(t("empty"))
else:
    listing_id = parse_id(query)
    if listing_id is None:
        st.warning(t("bad_input"))
    else:
        match = df[df["id"] == listing_id]
        if match.empty:
            st.warning(t("not_found", n=fnum(len(df))))
        else:
            render_result(match.iloc[0], split_ok)

st.write("")

# ---------------------------------------------------------------- tabs
tab_deals, tab_dist, tab_areas = st.tabs([t("tab_deals"), t("tab_dist"), t("tab_areas")])
cat_label = {k: t(f"cat_{k}") for k in VERDICT_ORDER}
cat_color = {cat_label[k]: COLOR[k] for k in VERDICT_ORDER}

with tab_deals:
    min_disc = st.slider(t("min_disc"), 0, 40, 10)
    deals = fdf[fdf["bargain_score"] >= min_disc].sort_values("bargain_score", ascending=False).head(100)
    if deals.empty:
        st.info(t("no_rows"))
    else:
        st.caption(t("shown", k=len(deals)))
        table = pd.DataFrame(
            {
                t("c_area"): deals["area_name"],
                t("c_rooms"): deals["rooms"].astype(int),
                t("c_size"): deals["area"],
                t("c_price"): deals["actual_price"],
                t("c_model"): deals["predicted_price"],
                t("c_score"): deals["bargain_score"],
                t("c_unseen"): deals["held_out"] if split_ok else False,
                t("c_link"): deals["url"],
            }
        )
        cfg = {
            t("c_size"): st.column_config.NumberColumn(format="%.0f"),
            t("c_price"): st.column_config.NumberColumn(format="localized"),
            t("c_model"): st.column_config.NumberColumn(format="localized"),
            t("c_score"): st.column_config.NumberColumn(format="%.1f"),
            t("c_link"): st.column_config.LinkColumn(display_text=t("link_text")),
        }
        if not split_ok:
            table = table.drop(columns=[t("c_unseen")])
        st.dataframe(table, hide_index=True, width="stretch", column_config=cfg, height=430)

with tab_dist:
    if fdf.empty:
        st.info(t("no_rows"))
    else:
        left, right = st.columns(2)
        plot = fdf.copy()
        plot["score_c"] = plot["bargain_score"].clip(-60, 60)
        plot["cat"] = plot["verdict"].map(cat_label)
        with left:
            st.markdown(f"**{t('dist_title')}**")
            fig = px.histogram(
                plot, x="score_c", color="cat", color_discrete_map=cat_color,
                category_orders={"cat": [cat_label[k] for k in VERDICT_ORDER]},
            )
            fig.update_traces(xbins=dict(start=-60, end=60, size=5))
            fig.update_layout(bargap=0.06, xaxis_title=t("x_score"), yaxis_title=t("y_count"))
            st.plotly_chart(chart_layout(fig), width="stretch")
            n_out = int((fdf["bargain_score"].abs() > 60).sum())
            if n_out:
                st.caption(t("clipped", n=n_out))
        with right:
            st.markdown(f"**{t('scatter_title')}**")
            plot["hover"] = (
                plot["area_name"] + "<br>"
                + plot["rooms"].astype(int).astype(str) + " / " + plot["area"].round(0).astype(int).astype(str) + " m²<br>"
                + plot["actual_price"].map(fnum) + " → " + plot["predicted_price"].map(fnum) + " AZN"
            )
            fig2 = px.scatter(
                plot, x="actual_price", y="predicted_price", color="cat", color_discrete_map=cat_color,
                category_orders={"cat": [cat_label[k] for k in VERDICT_ORDER]},
                custom_data=["hover"], log_x=True, log_y=True, opacity=0.75,
            )
            fig2.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker=dict(size=7))
            lo = float(min(plot["actual_price"].min(), plot["predicted_price"].min()))
            hi = float(max(plot["actual_price"].max(), plot["predicted_price"].max()))
            ticks, labels = price_ticks(lo, hi)
            for axis_update in (fig2.update_xaxes, fig2.update_yaxes):
                axis_update(tickmode="array", tickvals=ticks, ticktext=labels,
                            range=[math.log10(lo * 0.9), math.log10(hi * 1.1)])
            fig2.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name=t("equal"), hoverinfo="skip",
                           line=dict(color=MUTED, width=1, dash="dot"))
            )
            fig2.update_layout(xaxis_title=t("x_price"), yaxis_title=t("y_pred"))
            st.plotly_chart(chart_layout(fig2), width="stretch")
        if split_ok:
            held = df[df["held_out"]]
            st.caption(t("acc", n=n_held, mape=fpct(held["ape"].mean()), med=fpct(held["ape"].median())))

with tab_areas:
    MIN_N = 10
    grp = (
        fdf.groupby("area_name")
        .agg(n=("id", "size"), ppsm=("ppsm", "median"), cheap=("bargain_score", lambda s: int((s >= 10).sum())))
        .reset_index()
    )
    grp = grp[grp["n"] >= MIN_N].copy()
    grp["share"] = grp["cheap"] / grp["n"] * 100
    grp["label"] = [
        f"{fnum(sh, 0)}% ({int(c)}/{int(n_)})" for sh, c, n_ in zip(grp["share"], grp["cheap"], grp["n"])
    ]
    if grp.empty:
        st.info(t("area_none", k=MIN_N))
    else:
        st.caption(t("area_note", k=MIN_N))
        c1, c2 = st.columns(2)
        h = max(260, 26 * len(grp) + 60)
        with c1:
            st.markdown(f"**{t('area_ppsm')}**")
            f1 = px.bar(grp.sort_values("ppsm"), x="ppsm", y="area_name", orientation="h",
                        color="ppsm", color_continuous_scale=TEAL_SCALE)
            f1.update_coloraxes(showscale=False)
            f1.update_layout(xaxis_title=t("unit_sqm"), yaxis_title="")
            st.plotly_chart(chart_layout(f1, h), width="stretch")
        with c2:
            st.markdown(f"**{t('area_cheap')}**")
            f2 = px.bar(grp.sort_values("share"), x="share", y="area_name", orientation="h",
                        text="label", color="share", color_continuous_scale=TEAL_SCALE)
            f2.update_coloraxes(showscale=False)
            f2.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>")
            f2.update_xaxes(range=[0, float(grp["share"].max()) * 1.3])
            f2.update_layout(xaxis_title=t("x_share"), yaxis_title="")
            st.plotly_chart(chart_layout(f2, h), width="stretch")

st.caption(t("footer"))
