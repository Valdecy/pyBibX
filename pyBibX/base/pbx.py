############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# pyBibX - A Bibliometric and Scientometric Library

# Citation: 
# PEREIRA, V. (2022). Project: pyBibX, File: pbibx.py, GitHub repository: <https://github.com/Valdecy/pyBibX>

############################################################################

# Required Libraries
import networkx as nx             
import numpy as np                
import pandas as pd               
import plotly.graph_objects as go
import plotly.subplots as ps      
import plotly.io as pio           
import re                         
import squarify                  
import unicodedata                
import textwrap

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import stws

from bertopic import BERTopic                               
from collections import Counter
from difflib import SequenceMatcher
from matplotlib import pyplot as plt                       
plt.style.use('bmh')
#from scipy.spatial import ConvexHull   
from sentence_transformers import SentenceTransformer                    
from sklearn.cluster import KMeans                          
from sklearn.decomposition import TruncatedSVD as tsvd      
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  
from summarizer import Summarizer
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from umap import UMAP  
from wordcloud import WordCloud                             

############################################################################

# pbx Class
class pbx_probe():
    def __init__(self, file_bib, db = 'scopus', del_duplicated = True):
        self.data_base         =  db
        self.institution_names =  [ 'chuo kikuu', 'egyetemi', 'eyunivesithi', 'háskóli', 'inivèsite', 'inyuvesi', 'iunivesite',
                                    'jaamacad', "jami'a", 'kulanui', 'mahadum', 'oilthigh', 'ollscoile', 'oniversite', 'prifysgol',
                                    'sveučilište', 'unibersidad', 'unibertsitatea', 'univ', 'universidad', 'universidade',
                                    'universitas', 'universitat', 'universitate', 'universitato', 'universiteit', 'universitet',
                                    'universitetas', 'universiti', 'università', 'universität', 'université', 'universite',
                                    'universitāte', 'univerza', 'univerzita', 'univerzitet', 'univesithi', 'uniwersytet',
                                    'vniuersitatis', 'whare wananga', 'yliopisto', 'yunifasiti', 'yunivesite', 'yunivhesiti',
                                    'zanko', 'ülikool', 'üniversite', 'πανεπιστήμιο', 'универзитет', 'университет', 'універсітэт',
                                    'university', 'academy', 'institut', 'supérieur', 'ibmec', 'uff', 'gradevinski', 'lab.', 
                                    'politecnico', 'research', 'laborat', 'college'
                                  ]
        self.language_names  =    { 'afr': 'Afrikaans', 'alb': 'Albanian','amh': 'Amharic', 'ara': 'Arabic', 'arm': 'Armenian', 
                                    'aze': 'Azerbaijani', 'bos': 'Bosnian', 'bul': 'Bulgarian', 'cat': 'Catalan', 'chi': 'Chinese', 
                                    'cze': 'Czech', 'dan': 'Danish', 'dut': 'Dutch', 'eng': 'English', 'epo': 'Esperanto', 
                                    'est': 'Estonian', 'fin': 'Finnish', 'fre': 'French', 'geo': 'Georgian', 'ger': 'German', 
                                    'gla': 'Scottish Gaelic', 'gre': 'Greek, Modern', 'heb': 'Hebrew', 'hin': 'Hindi', 
                                    'hrv': 'Croatian', 'hun': 'Hungarian', 'ice': 'Icelandic', 'ind': 'Indonesian', 'ita': 'Italian', 
                                    'jpn': 'Japanese', 'kin': 'Kinyarwanda', 'kor': 'Korean', 'lat': 'Latin', 'lav': 'Latvian', 
                                    'lit': 'Lithuanian', 'mac': 'Macedonian', 'mal': 'Malayalam', 'mao': 'Maori', 'may': 'Malay', 
                                    'mul': 'Multiple languages', 'nor': 'Norwegian', 'per': 'Persian, Iranian', 'pol': 'Polish', 
                                    'por': 'Portuguese', 'pus': 'Pushto', 'rum': 'Romanian, Rumanian, Moldovan', 'rus': 'Russian', 
                                    'san': 'Sanskrit', 'slo': 'Slovak', 'slv': 'Slovenian', 'spa': 'Spanish', 'srp': 'Serbian', 
                                    'swe': 'Swedish', 'tha': 'Thai', 'tur': 'Turkish', 'ukr': 'Ukrainian', 'und': 'Undetermined', 
                                    'vie': 'Vietnamese', 'wel': 'Welsh'
                                  }
        self.country_names =      ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 
                                   'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 
                                   'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 
                                   'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire, Sint Eustatius and Saba', 
                                   'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 
                                   'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 
                                   'Canada', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 
                                   'Christmas Island', 'Cocos Islands', 'Colombia', 'Comoros', 'Democratic Republic of the Congo', 
                                   'Congo', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czechia', 
                                   "Côte d'Ivoire", 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 
                                   'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 
                                   'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 
                                   'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 
                                   'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 
                                   'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 
                                   'Holy See', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 
                                   'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 
                                   'Kenya', 'Kiribati', 'North Korea', 'South Korea', 'Kuwait', 'Kyrgyzstan', 
                                   "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 
                                   'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Madagascar', 'Malawi', 'Malaysia', 
                                   'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 
                                   'Mayotte', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 
                                   'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 
                                   'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 
                                   'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 
                                   'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 
                                   'Puerto Rico', 'Qatar', 'Republic of North Macedonia', 'Romania', 'Russian Federation', 'Rwanda', 
                                   'Réunion', 'Saint Barthelemy', 'Saint Helena, Ascension and Tristan da Cunha', 
                                   'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon', 
                                   'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 
                                   'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 
                                   'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 
                                   'South Georgia and the South Sandwich Islands', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 
                                   'Suriname', 'Svalbard and Jan Mayen', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 
                                   'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 
                                   'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 
                                   'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 
                                   'United States Minor Outlying Islands', 'United States of America', 'Uruguay', 'Uzbekistan', 
                                   'Vanuatu', 'Venezuela', 'Viet Nam', 'Virgin Islands (British)', 'Virgin Islands (U.S.)', 
                                   'Wallis and Futuna', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe', 'Aland Islands'
                                  ]
        self.country_alpha_2 =    ['AF', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 
                                   'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 
                                   'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 
                                   'KM', 'CD', 'CG', 'CK', 'CR', 'HR', 'CU', 'CW', 'CY', 'CZ', 'CI', 'DK', 'DJ', 'DM', 'DO', 'EC', 
                                   'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 
                                   'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 
                                   'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 
                                   'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 
                                   'LT', 'LU', 'MO', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 
                                   'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 
                                   'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 
                                   'PT', 'PR', 'QA', 'MK', 'RO', 'RU', 'RW', 'RE', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 
                                   'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 
                                   'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 
                                   'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VU', 'VE', 
                                   'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW', 'AX'
                                  ]
        self.country_alpha_3 =    ['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATA', 'ATG', 'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 
                                   'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BES', 'BIH', 
                                   'BWA', 'BVT', 'BRA', 'IOT', 'BRN', 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', 'CYM', 'CAF', 
                                   'TCD', 'CHL', 'CHN', 'CXR', 'CCK', 'COL', 'COM', 'COD', 'COG', 'COK', 'CRI', 'HRV', 'CUB', 'CUW', 
                                   'CYP', 'CZE', 'CIV', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 
                                   'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'GUF', 'PYF', 'ATF', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 
                                   'GIB', 'GRC', 'GRL', 'GRD', 'GLP', 'GUM', 'GTM', 'GGY', 'GIN', 'GNB', 'GUY', 'HTI', 'HMD', 'VAT', 
                                   'HND', 'HKG', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR', 'ITA', 'JAM', 'JPN', 
                                   'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 
                                   'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MTQ', 'MRT', 
                                   'MUS', 'MYT', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MSR', 'MAR', 'MOZ', 'MMR', 'NAM', 'NRU', 
                                   'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'NIU', 'NFK', 'MNP', 'NOR', 'OMN', 'PAK', 'PLW', 
                                   'PSE', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'PCN', 'POL', 'PRT', 'PRI', 'QAT', 'MKD', 'ROU', 'RUS', 
                                   'RWA', 'REU', 'BLM', 'SHN', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 
                                   'SRB', 'SYC', 'SLE', 'SGP', 'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SGS', 'SSD', 'ESP', 'LKA', 
                                   'SDN', 'SUR', 'SJM', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TKL', 'TON', 
                                   'TTO', 'TUN', 'TUR', 'TKM', 'TCA', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'UMI', 'USA', 'URY', 'UZB', 
                                   'VUT', 'VEN', 'VNM', 'VGB', 'VIR', 'WLF', 'ESH', 'YEM', 'ZMB', 'ZWE', 'ALA'
                                  ]
        self.country_numeric =    [4, 8, 12, 16, 20, 24, 660, 10, 28, 32, 51, 533, 36, 40, 31, 44, 48, 50, 52, 112, 56, 84, 204, 60, 
                                   64, 68, 535, 70, 72, 74, 76, 86, 96, 100, 854, 108, 132, 116, 120, 124, 136, 140, 148, 152, 156, 
                                   162, 166, 170, 174, 180, 178, 184, 188, 191, 192, 531, 196, 203, 384, 208, 262, 212, 214, 218, 818, 
                                   222, 226, 232, 233, 748, 231, 238, 234, 242, 246, 250, 254, 258, 260, 266, 270, 268, 276, 288, 292, 
                                   300, 304, 308, 312, 316, 320, 831, 324, 624, 328, 332, 334, 336, 340, 344, 348, 352, 356, 360, 364, 
                                   368, 372, 833, 376, 380, 388, 392, 832, 400, 398, 404, 296, 408, 410, 414, 417, 418, 428, 422, 426, 
                                   430, 434, 438, 440, 442, 446, 450, 454, 458, 462, 466, 470, 584, 474, 478, 480, 175, 484, 583, 498, 
                                   492, 496, 499, 500, 504, 508, 104, 516, 520, 524, 528, 540, 554, 558, 562, 566, 570, 574, 580, 578, 
                                   512, 586, 585, 275, 591, 598, 600, 604, 608, 612, 616, 620, 630, 634, 807, 642, 643, 646, 638, 652, 
                                   654, 659, 662, 663, 666, 670, 882, 674, 678, 682, 686, 688, 690, 694, 702, 534, 703, 705, 90, 706, 
                                   710, 239, 728, 724, 144, 729, 740, 744, 752, 756, 760, 158, 762, 834, 764, 626, 768, 772, 776, 780, 
                                   788, 792, 795, 796, 798, 800, 804, 784, 826, 581, 840, 858, 860, 548, 862, 704, 92, 850, 876, 732, 
                                   887, 894, 716, 248
                                  ] 
        self.country_lat_long =   [(33.93911, 67.709953), (41.153332, 20.168331), (28.033886, 1.659626), (-14.270972, -170.132217), 
                                   (42.546245, 1.601554), (-11.202692, 17.873887), (18.220554, -63.068615), (-75.250973, -0.071389), 
                                   (17.060816, -61.796428), (-38.416097, -63.616672), (40.069099, 45.038189), (12.52111, -69.968338), 
                                   (-25.274398, 133.775136), (47.516231, 14.550072), (40.143105, 47.576927), (25.03428, -77.39628), 
                                   (25.930414, 50.637772), (23.684994, 90.356331), (13.193887, -59.543198), (53.709807, 27.953389), 
                                   (50.503887, 4.469936), (17.189877, -88.49765), (9.30769, 2.315834), (32.321384, -64.75737), 
                                   (27.514162, 90.433601), (-16.290154, -63.588653), (12.15, -68.26667), (43.915886, 17.679076), 
                                   (-22.328474, 24.684866), (-54.423199, 3.413194), (-14.235004, -51.92528), (-6.343194, 71.876519), 
                                   (4.535277, 114.727669), (42.733883, 25.48583), (12.238333, -1.561593), (-3.373056, 29.918886), 
                                   (16.002082, -24.013197), (12.565679, 104.990963), (7.369722, 12.354722), (56.130366, -106.346771), 
                                   (19.513469, -80.566956), (6.611111, 20.939444), (15.454166, 18.732207), (-35.675147, -71.542969), 
                                   (35.86166, 104.195397), (-10.447525, 105.690449), (-12.164165, 96.870956), (4.570868, -74.297333), 
                                   (-11.875001, 43.872219), (-4.038333, 21.758664), (-0.228021, 15.827659), (-21.236736, -159.777671), 
                                   (9.748917, -83.753428), (45.1, 15.2), (21.521757, -77.781167), (12.16957, -68.990021), 
                                   (35.126413, 33.429859), (49.817492, 15.472962), (7.539989, -5.54708), (56.26392, 9.501785), 
                                   (11.825138, 42.590275), (15.414999, -61.370976), (18.735693, -70.162651), (-1.831239, -78.183406), 
                                   (26.820553, 30.802498), (13.794185, -88.89653), (1.650801, 10.267895), (15.179384, 39.782334), 
                                   (58.595272, 25.013607), (-26.522503, 31.465866), (9.145, 40.489673), (-51.796253, -59.523613), 
                                   (61.892635, -6.911806), (-16.578193, 179.414413), (61.92411, 25.748151), (46.227638, 2.213749), 
                                   (3.933889, -53.125782), (-17.679742, -149.406843), (-49.280366, 69.348557), (-0.803689, 11.609444), 
                                   (13.443182, -15.310139), (42.315407, 43.356892), (51.165691, 10.451526), (7.946527, -1.023194), 
                                   (36.137741, -5.345374), (39.074208, 21.824312), (71.706936, -42.604303), (12.262776, -61.604171), 
                                   (16.995971, -62.067641), (13.444304, 144.793731), (15.783471, -90.230759), (49.465691, -2.585278), 
                                   (9.945587, -9.696645), (11.803749, -15.180413), (4.860416, -58.93018), (18.971187, -72.285215), 
                                   (-53.08181, 73.504158), (41.902916, 12.453389), (15.199999, -86.241905), (22.396428, 114.109497), 
                                   (47.162494, 19.503304), (64.963051, -19.020835), (20.593684, 78.96288), (-0.789275, 113.921327), 
                                   (32.427908, 53.688046), (33.223191, 43.679291), (53.41291, -8.24389), (54.236107, -4.548056), 
                                   (31.046051, 34.851612), (41.87194, 12.56738), (18.109581, -77.297508), (36.204824, 138.252924), 
                                   (49.214439, -2.13125), (30.585164, 36.238414), (48.019573, 66.923684), (-0.023559, 37.906193), 
                                   (-3.370417, -168.734039), (40.339852, 127.510093), (35.907757, 127.766922), (29.31166, 47.481766), 
                                   (41.20438, 74.766098), (19.85627, 102.495496), (56.879635, 24.603189), (33.854721, 35.862285), 
                                   (-29.609988, 28.233608), (6.428055, -9.429499), (26.3351, 17.228331), (47.166, 9.555373), 
                                   (55.169438, 23.881275), (49.815273, 6.129583), (22.198745, 113.543873), (-18.766947, 46.869107), 
                                   (-13.254308, 34.301525), (4.210484, 101.975766), (3.202778, 73.22068), (17.570692, -3.996166), 
                                   (35.937496, 14.375416), (7.131474, 171.184478), (14.641528, -61.024174), (21.00789, -10.940835), 
                                   (-20.348404, 57.552152), (-12.8275, 45.166244), (23.634501, -102.552784), (7.425554, 150.550812), 
                                   (47.411631, 28.369885), (43.750298, 7.412841), (46.862496, 103.846656), (42.708678, 19.37439), 
                                   (16.742498, -62.187366), (31.791702, -7.09262), (-18.665695, 35.529562), (21.913965, 95.956223), 
                                   (-22.95764, 18.49041), (-0.522778, 166.931503), (28.394857, 84.124008), (52.132633, 5.291266), 
                                   (-20.904305, 165.618042), (-40.900557, 174.885971), (12.865416, -85.207229), (17.607789, 8.081666), 
                                   (9.081999, 8.675277), (-19.054445, -169.867233), (-29.040835, 167.954712), (17.33083, 145.38469), 
                                   (60.472024, 8.468946), (21.512583, 55.923255), (30.375321, 69.345116), (7.51498, 134.58252), 
                                   (31.952162, 35.233154), (8.537981, -80.782127), (-6.314993, 143.95555), (-23.442503, -58.443832), 
                                   (-9.189967, -75.015152), (12.879721, 121.774017), (-24.703615, -127.439308), 
                                   (51.919438, 19.145136), (39.399872, -8.224454), (18.220833, -66.590149), (25.354826, 51.183884), 
                                   (41.608635, 21.745275), (45.943161, 24.96676), (61.52401, 105.318756), (-1.940278, 29.873888), 
                                   (-21.115141, 55.536384), (17.9, 62.8333), (-24.143474, -10.030696), (17.357822, -62.782998), 
                                   (13.909444, -60.978893), (18.073099, -63.082199), (46.941936, -56.27111), (12.984305, -61.287228), 
                                   (-13.759029, -172.104629), (43.94236, 12.457777), (0.18636, 6.613081), (23.885942, 45.079162), 
                                   (14.497401, -14.452362), (44.016521, 21.005859), (-4.679574, 55.491977), (8.460555, -11.779889), 
                                   (1.352083, 103.819836), (18.0425, 63.0548), (48.669026, 19.699024), (46.151241, 14.995463), 
                                   (-9.64571, 160.156194), (5.152149, 46.199616), (-30.559482, 22.937506), (-54.429579, -36.587909), 
                                   (6.877, 31.307), (40.463667, -3.74922), (7.873054, 80.771797), (12.862807, 30.217636), 
                                   (3.919305, -56.027783), (77.553604, 23.670272), (60.128161, 18.643501), (46.818188, 8.227512), 
                                   (34.802075, 38.996815), (23.69781, 120.960515), (38.861034, 71.276093), (-6.369028, 34.888822), 
                                   (15.870032, 100.992541), (-8.874217, 125.727539), (8.619543, 0.824782), (-8.967363, -171.855881), 
                                   (-21.178986, -175.198242), (10.691803, -61.222503), (33.886917, 9.537499), (38.963745, 35.243322), 
                                   (38.969719, 59.556278), (21.694025, -71.797928), (-7.109535, 177.64933), (1.373333, 32.290275), 
                                   (48.379433, 31.16558), (23.424076, 53.847818), (55.378051, -3.435973), (19.2823, 166.647), 
                                   (37.09024, -95.712891), (-32.522779, -55.765835), (41.377491, 64.585262), (-15.376706, 166.959158), 
                                   (6.42375, -66.58973), (14.058324, 108.277199), (18.420695, -64.639968), (18.335765, -64.896335), 
                                   (-13.768752, -177.156097), (24.215527, -12.885834), (15.552727, 48.516388), 
                                   (-13.133897, 27.849332), (-19.015438, 29.154857), (60.1785, 19.9156)
                                  ]
        self.color_names =        [ '#6929c4', '#9f1853', '#198038', '#b28600', '#8a3800', '#1192e8', '#fa4d56', '#002d9c', 
                                    '#009d9a', '#a56eff', '#005d5d', '#570408', '#ee538b', '#012749', '#da1e28', '#f1c21b', 
                                    '#ff832b', '#198038', '#bdd9bf', '#929084', '#ffc857', '#a997df', '#e5323b', '#2e4052', 
                                    '#e1daae', '#ff934f', '#cc2d35', '#058ed9', '#848fa2', '#2d3142', '#62a3f0', '#cc5f54', 
                                    '#e6cb60', '#523d02', '#c67ce6', '#00b524', '#4ad9bd', '#f53347', '#565c55',
                                    '#000000', '#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059',
                                    '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87',
                                    '#5a0007', '#809693', '#feffe6', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80',
                                    '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa', '#d16100',
                                    '#ddefff', '#000035', '#7b4f4b', '#a1c299', '#300018', '#0aa6d8', '#013349', '#00846f',
                                    '#372101', '#ffb500', '#c2ffed', '#a079bf', '#cc0744', '#c0b9b2', '#c2ff99', '#001e09',
                                    '#00489c', '#6f0062', '#0cbd66', '#eec3ff', '#456d75', '#b77b68', '#7a87a1', '#788d66',
                                    '#885578', '#fad09f', '#ff8a9a', '#d157a0', '#bec459', '#456648', '#0086ed', '#886f4c',
                                    '#34362d', '#b4a8bd', '#00a6aa', '#452c2c', '#636375', '#a3c8c9', '#ff913f', '#938a81',
                                    '#575329', '#00fecf', '#b05b6f', '#8cd0ff', '#3b9700', '#04f757', '#c8a1a1', '#1e6e00',
                                    '#7900d7', '#a77500', '#6367a9', '#a05837', '#6b002c', '#772600', '#d790ff', '#9b9700',
                                    '#549e79', '#fff69f', '#201625', '#72418f', '#bc23ff', '#99adc0', '#3a2465', '#922329',
                                    '#5b4534', '#fde8dc', '#404e55', '#0089a3', '#cb7e98', '#a4e804', '#324e72', '#6a3a4c',
                                    '#83ab58', '#001c1e', '#d1f7ce', '#004b28', '#c8d0f6', '#a3a489', '#806c66', '#222800',
                                    '#bf5650', '#e83000', '#66796d', '#da007c', '#ff1a59', '#8adbb4', '#1e0200', '#5b4e51',
                                    '#c895c5', '#320033', '#ff6832', '#66e1d3', '#cfcdac', '#d0ac94', '#7ed379', '#012c58',
                                    '#7a7bff', '#d68e01', '#353339', '#78afa1', '#feb2c6', '#75797c', '#837393', '#943a4d',
                                    '#b5f4ff', '#d2dcd5', '#9556bd', '#6a714a', '#001325', '#02525f', '#0aa3f7', '#e98176',
                                    '#dbd5dd', '#5ebcd1', '#3d4f44', '#7e6405', '#02684e', '#962b75', '#8d8546', '#9695c5',
                                    '#e773ce', '#d86a78', '#3e89be', '#ca834e', '#518a87', '#5b113c', '#55813b', '#e704c4',
                                    '#00005f', '#a97399', '#4b8160', '#59738a', '#ff5da7', '#f7c9bf', '#643127', '#513a01',
                                    '#6b94aa', '#51a058', '#a45b02', '#1d1702', '#e20027', '#e7ab63', '#4c6001', '#9c6966',
                                    '#64547b', '#97979e', '#006a66', '#391406', '#f4d749', '#0045d2', '#006c31', '#ddb6d0',
                                    '#7c6571', '#9fb2a4', '#00d891', '#15a08a', '#bc65e9', '#fffffe', '#c6dc99', '#203b3c',
                                    '#671190', '#6b3a64', '#f5e1ff', '#ffa0f2', '#ccaa35', '#374527', '#8bb400', '#797868',
                                    '#c6005a', '#3b000a', '#c86240', '#29607c', '#402334', '#7d5a44', '#ccb87c', '#b88183',
                                    '#aa5199', '#b5d6c3', '#a38469', '#9f94f0', '#a74571', '#b894a6', '#71bb8c', '#00b433',
                                    '#789ec9', '#6d80ba', '#953f00', '#5eff03', '#e4fffc', '#1be177', '#bcb1e5', '#76912f',
                                    '#003109', '#0060cd', '#d20096', '#895563', '#29201d', '#5b3213', '#a76f42', '#89412e',
                                    '#1a3a2a', '#494b5a', '#a88c85', '#f4abaa', '#a3f3ab', '#00c6c8', '#ea8b66', '#958a9f',
                                    '#bdc9d2', '#9fa064', '#be4700', '#658188', '#83a485', '#453c23', '#47675d', '#3a3f00',
                                    '#061203', '#dffb71', '#868e7e', '#98d058', '#6c8f7d', '#d7bfc2', '#3c3e6e', '#d83d66',
                                    '#2f5d9b', '#6c5e46', '#d25b88', '#5b656c', '#00b57f', '#545c46', '#866097', '#365d25',
                                    '#252f99', '#00ccff', '#674e60', '#fc009c', '#92896b', '#1e2324', '#dec9b2', '#9d4948',
                                    '#85abb4', '#342142', '#d09685', '#a4acac', '#00ffff', '#ae9c86', '#742a33', '#0e72c5',
                                    '#afd8ec', '#c064b9', '#91028c', '#feedbf', '#ffb789', '#9cb8e4', '#afffd1', '#2a364c',
                                    '#4f4a43', '#647095', '#34bbff', '#807781', '#920003', '#b3a5a7', '#018615', '#f1ffc8',
                                    '#976f5c', '#ff3bc1', '#ff5f6b', '#077d84', '#f56d93', '#5771da', '#4e1e2a', '#830055',
                                    '#02d346', '#be452d', '#00905e', '#be0028', '#6e96e3', '#007699', '#fec96d', '#9c6a7d',
                                    '#3fa1b8', '#893de3', '#79b4d6', '#7fd4d9', '#6751bb', '#b28d2d', '#e27a05', '#dd9cb8',
                                    '#aabc7a', '#980034', '#561a02', '#8f7f00', '#635000', '#cd7dae', '#8a5e2d', '#ffb3e1',
                                    '#6b6466', '#c6d300', '#0100e2', '#88ec69', '#8fccbe', '#21001c', '#511f4d', '#e3f6e3',
                                    '#ff8eb1', '#6b4f29', '#a37f46', '#6a5950', '#1f2a1a', '#04784d', '#101835', '#e6e0d0',
                                    '#ff74fe', '#00a45f', '#8f5df8', '#4b0059', '#412f23', '#d8939e', '#db9d72', '#604143',
                                    '#b5bace', '#989eb7', '#d2c4db', '#a587af', '#77d796', '#7f8c94', '#ff9b03', '#555196',
                                    '#31ddae', '#74b671', '#802647', '#2a373f', '#014a68', '#696628', '#4c7b6d', '#002c27',
                                    '#7a4522', '#3b5859', '#e5d381', '#fff3ff', '#679fa0', '#261300', '#2c5742', '#9131af',
                                    '#af5d88', '#c7706a', '#61ab1f', '#8cf2d4', '#c5d9b8', '#9ffffb', '#bf45cc', '#493941',
                                    '#863b60', '#b90076', '#003177', '#c582d2', '#c1b394', '#602b70', '#887868', '#babfb0',
                                    '#030012', '#d1acfe', '#7fdefe', '#4b5c71', '#a3a097', '#e66d53', '#637b5d', '#92bea5',
                                    '#00f8b3', '#beddff', '#3db5a7', '#dd3248', '#b6e4de', '#427745', '#598c5a', '#b94c59',
                                    '#8181d5', '#94888b', '#fed6bd', '#536d31', '#6eff92', '#e4e8ff', '#20e200', '#ffd0f2',
                                    '#4c83a1', '#bd7322', '#915c4e', '#8c4787', '#025117', '#a2aa45', '#2d1b21', '#a9ddb0',
                                    '#ff4f78', '#528500', '#009a2e', '#17fce4', '#71555a', '#525d82', '#00195a', '#967874',
                                    '#555558', '#0b212c', '#1e202b', '#efbfc4', '#6f9755', '#6f7586', '#501d1d', '#372d00',
                                    '#741d16', '#5eb393', '#b5b400', '#dd4a38', '#363dff', '#ad6552', '#6635af', '#836bba',
                                    '#98aa7f', '#464836', '#322c3e', '#7cb9ba', '#5b6965', '#707d3d', '#7a001d', '#6e4636',
                                    '#443a38', '#ae81ff', '#489079', '#897334', '#009087', '#da713c', '#361618', '#ff6f01',
                                    '#006679', '#370e77', '#4b3a83', '#c9e2e6', '#c44170', '#ff4526', '#73be54', '#c4df72',
                                    '#adff60', '#00447d', '#dccec9', '#bd9479', '#656e5b', '#ec5200', '#ff6ec2', '#7a617e',
                                    '#ddaea2', '#77837f', '#a53327', '#608eff', '#b599d7', '#a50149', '#4e0025', '#c9b1a9',
                                    '#03919a', '#1b2a25', '#e500f1', '#982e0b', '#b67180', '#e05859', '#006039', '#578f9b',
                                    '#305230', '#ce934c', '#b3c2be', '#c0bac0', '#b506d3', '#170c10', '#4c534f', '#224451',
                                    '#3e4141', '#78726d', '#b6602b', '#200441', '#ddb588', '#497200', '#c5aab6', '#033c61',
                                    '#71b2f5', '#a9e088', '#4979b0', '#a2c3df', '#784149', '#2d2b17', '#3e0e2f', '#57344c',
                                    '#0091be', '#e451d1', '#4b4b6a', '#5c011a', '#7c8060', '#ff9491', '#4c325d', '#005c8b',
                                    '#e5fda4', '#68d1b6', '#032641', '#140023', '#8683a9', '#cfff00', '#a72c3e', '#34475a',
                                    '#b1bb9a', '#b4a04f', '#8d918e', '#a168a6', '#813d3a', '#425218', '#da8386', '#776133',
                                    '#563930', '#8498ae', '#90c1d3', '#b5666b', '#9b585e', '#856465', '#ad7c90', '#e2bc00',
                                    '#e3aae0', '#b2c2fe', '#fd0039', '#009b75', '#fff46d', '#e87eac', '#dfe3e6', '#848590',
                                    '#aa9297', '#83a193', '#577977', '#3e7158', '#c64289', '#ea0072', '#c4a8cb', '#55c899',
                                    '#e78fcf', '#004547', '#f6e2e3', '#966716', '#378fdb', '#435e6a', '#da0004', '#1b000f',
                                    '#5b9c8f', '#6e2b52', '#011115', '#e3e8c4', '#ae3b85', '#ea1ca9', '#ff9e6b', '#457d8b',
                                    '#92678b', '#00cdbb', '#9ccc04', '#002e38', '#96c57f', '#cff6b4', '#492818', '#766e52',
                                    '#20370e', '#e3d19f', '#2e3c30', '#b2eace', '#f3bda4', '#a24e3d', '#976fd9', '#8c9fa8',
                                    '#7c2b73', '#4e5f37', '#5d5462', '#90956f', '#6aa776', '#dbcbf6', '#da71ff', '#987c95',
                                    '#52323c', '#bb3c42', '#584d39', '#4fc15f', '#a2b9c1', '#79db21', '#1d5958', '#bd744e',
                                    '#160b00', '#20221a', '#6b8295', '#00e0e4', '#102401', '#1b782a', '#daa9b5', '#b0415d',
                                    '#859253', '#97a094', '#06e3c4', '#47688c', '#7c6755', '#075c00', '#7560d5', '#7d9f00',
                                    '#c36d96', '#4d913e', '#5f4276', '#fce4c8', '#303052', '#4f381b', '#e5a532', '#706690',
                                    '#aa9a92', '#237363', '#73013e', '#ff9079', '#a79a74', '#029bdb', '#ff0169', '#c7d2e7',
                                    '#ca8869', '#80ffcd', '#bb1f69', '#90b0ab', '#7d74a9', '#fcc7db', '#99375b', '#00ab4d',
                                    '#abaed1', '#be9d91', '#e6e5a7', '#332c22', '#dd587b', '#f5fff7', '#5d3033', '#6d3800',
                                    '#ff0020', '#b57bb3', '#d7ffe6', '#c535a9', '#260009', '#6a8781', '#a8abb4', '#d45262',
                                    '#794b61', '#4621b2', '#8da4db', '#c7c890', '#6fe9ad', '#a243a7', '#b2b081', '#181b00',
                                    '#286154', '#4ca43b', '#6a9573', '#a8441d', '#5c727b', '#738671', '#d0cfcb', '#897b77',
                                    '#1f3f22', '#4145a7', '#da9894', '#a1757a', '#63243c', '#adaaff', '#00cde2', '#ddbc62',
                                    '#698eb1', '#208462', '#00b7e0', '#614a44', '#9bbb57', '#7a5c54', '#857a50', '#766b7e',
                                    '#014833', '#ff8347', '#7a8eba', '#274740', '#946444', '#ebd8e6', '#646241', '#373917',
                                    '#6ad450', '#81817b', '#d499e3', '#979440', '#011a12', '#526554', '#b5885c', '#a499a5',
                                    '#03ad89', '#b3008b', '#e3c4b5', '#96531f', '#867175', '#74569e', '#617d9f', '#e70452',
                                    '#067eaf', '#a697b6', '#b787a8', '#9cff93', '#311d19', '#3a9459', '#6e746e', '#b0c5ae',
                                    '#84edf7', '#ed3488', '#754c78', '#384644', '#c7847b', '#00b6c5', '#7fa670', '#c1af9e',
                                    '#2a7fff', '#72a58c', '#ffc07f', '#9debdd', '#d97c8e', '#7e7c93', '#62e674', '#b5639e',
                                    '#ffa861', '#c2a580', '#8d9c83', '#b70546', '#372b2e', '#0098ff', '#985975', '#20204c',
                                    '#ff6c60', '#445083', '#8502aa', '#72361f', '#9676a3', '#484449', '#ced6c2', '#3b164a',
                                    '#cca763', '#2c7f77', '#02227b', '#a37e6f', '#cde6dc', '#cdfffb', '#be811a', '#f77183',
                                    '#ede6e2', '#cdc6b4', '#ffe09e', '#3a7271', '#ff7b59', '#4e4e01', '#4ac684', '#8bc891',
                                    '#bc8a96', '#cf6353', '#dcde5c', '#5eaadd', '#f6a0ad', '#e269aa', '#a3dae4', '#436e83',
                                    '#002e17', '#ecfbff', '#a1c2b6', '#50003f', '#71695b', '#67c4bb', '#536eff', '#5d5a48',
                                    '#890039', '#969381', '#371521', '#5e4665', '#aa62c3', '#8d6f81', '#2c6135', '#410601',
                                    '#564620', '#e69034', '#6da6bd', '#e58e56', '#e3a68b', '#48b176', '#d27d67', '#b5b268',
                                    '#7f8427', '#ff84e6', '#435740', '#eae408', '#f4f5ff', '#325800', '#4b6ba5', '#adceff',
                                    '#9b8acc', '#885138', '#5875c1', '#7e7311', '#fea5ca', '#9f8b5b', '#a55b54', '#89006a',
                                    '#af756f', '#2a2000', '#576e4a', '#7f9eff', '#7499a1', '#ffb550', '#00011e', '#d1511c',
                                    '#688151', '#bc908a', '#78c8eb', '#8502ff', '#483d30', '#c42221', '#5ea7ff', '#785715',
                                    '#0cea91', '#fffaed', '#b3af9d', '#3e3d52', '#5a9bc2', '#9c2f90', '#8d5700', '#add79c',
                                    '#00768b', '#337d00', '#c59700', '#3156dc', '#944575', '#ecffdc', '#d24cb2', '#97703c',
                                    '#4c257f', '#9e0366', '#88ffec', '#b56481', '#396d2b', '#56735f', '#988376', '#9bb195',
                                    '#a9795c', '#e4c5d3', '#9f4f67', '#1e2b39', '#664327', '#afce78', '#322edf', '#86b487',
                                    '#c23000', '#abe86b', '#96656d', '#250e35', '#a60019', '#0080cf', '#caefff', '#323f61',
                                    '#a449dc', '#6a9d3b', '#ff5ae4', '#636a01', '#d16cda', '#736060', '#ffbaad', '#d369b4',
                                    '#ffded6', '#6c6d74', '#927d5e', '#845d70', '#5b62c1', '#2f4a36', '#e45f35', '#ff3b53',
                                    '#ac84dd', '#762988', '#70ec98', '#408543', '#2c3533', '#2e182d', '#323925', '#19181b',
                                    '#2f2e2c', '#023c32', '#9b9ee2', '#58afad', '#5c424d', '#7ac5a6', '#685d75', '#b9bcbd',
                                    '#834357', '#1a7b42', '#2e57aa', '#e55199', '#316e47', '#cd00c5', '#6a004d', '#7fbbec',
                                    '#f35691', '#d7c54a', '#62acb7', '#cba1bc', '#a28a9a', '#6c3f3b', '#ffe47d', '#dcbae3',
                                    '#5f816d', '#3a404a', '#7dbf32', '#e6ecdc', '#852c19', '#285366', '#b8cb9c', '#0e0d00',
                                    '#4b5d56', '#6b543f', '#e27172', '#0568ec', '#2eb500', '#d21656', '#efafff', '#682021',
                                    '#2d2011', '#da4cff', '#70968e', '#ff7b7d', '#4a1930', '#e8c282', '#e7dbbc', '#a68486',
                                    '#1f263c', '#36574e', '#52ce79', '#adaaa9', '#8a9f45', '#6542d2', '#00fb8c', '#5d697b',
                                    '#ccd27f', '#94a5a1', '#790229', '#e383e6', '#7ea4c1', '#4e4452', '#4b2c00', '#620b70',
                                    '#314c1e', '#874aa6', '#e30091', '#66460a', '#eb9a8b', '#eac3a3', '#98eab3', '#ab9180',
                                    '#b8552f', '#1a2b2f', '#94ddc5', '#9d8c76', '#9c8333', '#94a9c9', '#392935', '#8c675e',
                                    '#cce93a', '#917100', '#01400b', '#449896', '#1ca370', '#e08da7', '#8b4a4e', '#667776',
                                    '#4692ad', '#67bda8', '#69255c', '#d3bfff', '#4a5132', '#7e9285', '#77733c', '#e7a0cc',
                                    '#51a288', '#2c656a', '#4d5c5e', '#c9403a', '#ddd7f3', '#005844', '#b4a200', '#488f69',
                                    '#858182', '#d4e9b9', '#3d7397', '#cae8ce', '#d60034', '#aa6746', '#9e5585', '#ba6200',
                                    '#dee3E9', '#ebbaB5', '#fef3c7', '#a6e3d7', '#cbb4d5', '#808b96', '#f7dc6f', '#48c9b0',
                                    '#af7ac5', '#ec7063', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53',
                                    '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16',
                                    '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80',
                                    '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'
                                  ]
        self.data, self.entries = self.__read_bib(file_bib, db, del_duplicated)
        self.__make_bib()
    
    # Function: Prepare .bib File
    def __make_bib(self, verbose = True):
        self.dy                 = pd.to_numeric(self.data['year'], downcast = 'float')
        self.date_str           = int(self.dy.min())
        self.date_end           = int(self.dy.max())
        self.doc_types          = self.data['document_type'].value_counts().sort_index()
        self.av_d_year          = self.dy.value_counts().sort_index()
        self.av_d_year          = round(self.av_d_year.mean(), 2)
        self.citation           = self.__get_citations(self.data['note'])
        self.av_c_doc           = round(sum(self.citation)/self.data.shape[0], 2)
        self.ref, self.u_ref    = self.__get_str(entry = 'references', s = ';',     lower = False, sorting = True)
        self.aut, self.u_aut    = self.__get_str(entry = 'author',     s = ' and ', lower = True,  sorting = True)
        self.aut_h              = self.__h_index()
        self.aut_docs           = [len(item) for item in self.aut]
        self.aut_single         = len([item  for item in self.aut_docs if item == 1])
        self.aut_multi          = [item for item in self.aut_docs if item > 1]
        self.aut_cit            = self.__get_counts(self.u_aut, self.aut, self.citation)
        self.kid, self.u_kid    = self.__get_str(entry = 'keywords', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_kid):
            self.u_kid.remove('unknow')
        self.kid_               = [item for sublist in self.kid for item in sublist]
        self.kid_count          = [self.kid_.count(item) for item in self.u_kid]
        idx                     = sorted(range(len(self.kid_count)), key = self.kid_count.__getitem__)
        idx.reverse()
        self.u_kid              = [self.u_kid[i] for i in idx]
        self.kid_count          = [self.kid_count[i] for i in idx]
        self.auk, self.u_auk    = self.__get_str(entry = 'author_keywords', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_auk):
            self.u_auk.remove('unknow')
        self.auk_               = [item for sublist in self.auk for item in sublist]
        self.auk_count          = [self.auk_.count(item) for item in self.u_auk]
        idx                     = sorted(range(len(self.auk_count)), key = self.auk_count.__getitem__)
        idx.reverse()
        self.u_auk              = [self.u_auk[i] for i in idx]
        self.auk_count          = [self.auk_count[i] for i in idx]
        self.jou, self.u_jou    = self.__get_str(entry = 'abbrev_source_title', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_jou):
            self.u_jou.remove('unknow')
        jou_                    = [item for sublist in self.jou for item in sublist]
        self.jou_count          = [jou_.count(item) for item in self.u_jou]
        idx                     = sorted(range(len(self.jou_count)), key = self.jou_count.__getitem__)
        idx.reverse()
        self.u_jou              = [self.u_jou[i] for i in idx]
        self.jou_count          = [self.jou_count[i] for i in idx]
        self.jou_cit            = self.__get_counts(self.u_jou, self.jou, self.citation)
        self.jou_cit            = self.__get_counts(self.u_jou, self.jou, self.citation)
        self.lan, self.u_lan    = self.__get_str(entry = 'language', s = '.', lower = True, sorting = True) 
        lan_                    = [item for sublist in self.lan for item in sublist]
        self.lan_count          = [lan_.count(item) for item in self.u_lan]
        self.ctr, self.u_ctr    = self.__get_countries()
        ctr_                    = [self.ctr[i][j] for i in range(0, len(self.aut)) for j in range(0, len(self.aut[i]))]
        self.ctr_count          = [ctr_.count(item) for item in self.u_ctr]
        self.ctr_cit            = self.__get_counts(self.u_ctr, self.ctr, self.citation)
        self.uni, self.u_uni    = self.__get_institutions() 
        uni_                    = [item for sublist in self.uni for item in sublist]
        self.uni_count          = [uni_.count(item) for item in self.u_uni]
        self.uni_cit            = self.__get_counts(self.u_uni,self.uni, self.citation)
        self.doc_aut            = self.__get_counts(self.u_aut, self.aut)
        self.av_doc_aut         = round(sum(self.doc_aut)/len(self.doc_aut), 2)
        self.t_c, self.s_c      = self.__total_and_self_citations()
        self.dy_ref             = self.__get_ref_year()
        self.natsort            = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]  
        #self.ordinal            = lambda n: '%d%s'%(n, {1: 'st', 2: 'nd', 3: 'rd'}.get(n if n < 20 else n % 10, 'th')) # [ordinal(n) for n in range(1, 15)]
        self.dy_c_year          = self.__get_collaboration_year()
        if ('UNKNOW' in self.u_ref):
            self.u_ref.remove('UNKNOW')
        self.__id_document()
        self.__id_author()
        self.__id_source()
        self.__id_institution()
        self.__id_country()
        self.__id_kwa()
        self.__id_kwp()
        if (verbose == True):
            for i in range(0, len(self.vb)):
                print(self.vb[i])
        return
    
    # Function: Document ID
    def __id_document(self):
        doc_list          = [str(i) for i in range(0, self.data.shape[0])]
        docs              = [self.data.loc[i, 'author']+' ('+self.data.loc[i, 'year']+'). '+self.data.loc[i, 'title']+'. '+self.data.loc[i, 'journal']+'. doi:'+self.data.loc[i, 'doi']+'. ' for i in range(0, self.data.shape[0])]
        self.table_id_doc = pd.DataFrame(zip(doc_list, docs), columns = ['ID', 'Document'])
        self.dict_id_doc  = dict(zip(doc_list, docs))
        return
    
    # Function: Author ID
    def __id_author(self):
        aut_list          = ['a_'+str(i) for i in range(0, len(self.u_aut))]
        self.table_id_aut = pd.DataFrame(zip(aut_list, self.u_aut), columns = ['ID', 'Author'])
        self.dict_id_aut  = dict(zip(aut_list, self.u_aut))
        self.dict_aut_id  = dict(zip(self.u_aut, aut_list))
        return
    
    # Function: Source ID
    def __id_source(self):
        jou_list          = ['j_'+str(i) for i in range(0, len(self.u_jou))]
        self.table_id_jou = pd.DataFrame(zip(jou_list, self.u_jou), columns = ['ID', 'Source'])
        self.dict_id_jou  = dict(zip(jou_list, self.u_jou))
        self.dict_jou_id  = dict(zip(self.u_jou, jou_list))
        return
    
    # Function: Institution ID
    def __id_institution(self):
        uni_list          = ['i_'+str(i) for i in range(0, len(self.u_uni))]
        self.table_id_uni = pd.DataFrame(zip(uni_list, self.u_uni), columns = ['ID', 'Institution'])
        self.dict_id_uni  = dict(zip(uni_list, self.u_uni))
        self.dict_uni_id  = dict(zip(self.u_uni, uni_list))
        return
    
    # Function: Country ID
    def __id_country(self):
        ctr_list          = ['c_'+str(i) for i in range(0, len(self.u_ctr))]
        self.table_id_ctr = pd.DataFrame(zip(ctr_list, self.u_ctr), columns = ['ID', 'Country'])
        self.dict_id_ctr  = dict(zip(ctr_list, self.u_ctr))
        self.dict_ctr_id  = dict(zip(self.u_ctr, ctr_list))
        return
    
    # Function: Authors' Keyword ID
    def __id_kwa(self):
        kwa_list          = ['k_'+str(i) for i in range(0, len(self.u_auk))]
        self.table_id_kwa = pd.DataFrame(zip(kwa_list, self.u_auk), columns = ['ID', 'KWA'])
        self.dict_id_kwa  = dict(zip(kwa_list, self.u_auk))
        self.dict_kwa_id  = dict(zip(self.u_auk, kwa_list))
        return
    
    # Function: Keywords Plus ID
    def __id_kwp(self):
        kwp_list          = ['p_'+str(i) for i in range(0, len(self.u_kid))]
        self.table_id_kwp = pd.DataFrame(zip(kwp_list, self.u_kid), columns = ['ID', 'KWP'])
        self.dict_id_kwp  = dict(zip(kwp_list, self.u_kid))
        self.dict_kwp_id  = dict(zip(self.u_kid, kwp_list))
        return
    
    # Function: ID types
    def id_doc_types(self):
        dt     = self.doc_types.index.to_list()
        dt_ids = []
        for i in range(0, len(dt)):
            item = dt[i]
            idx  = self.data.index[self.data['document_type'] == item].tolist()
            dt_ids.append([item, idx])
        report_dt = pd.DataFrame(dt_ids, columns = ['Document Types', 'IDs'])
        return report_dt

    # Function: Filter
    def filter_bib(self, doc_type = [], year_str = -1, year_end = -1, sources = [], core = -1, country = [], language = [], abstract = False):
        docs = []
        if (len(doc_type) > 0):
            for item in doc_type:
                if (sum(self.data['document_type'].isin([item])) > 0):
                    docs.append(item) 
                    self.data = self.data[self.data['document_type'].isin(docs)]
                    self.data = self.data.reset_index(drop = True)
                    self.__make_bib(verbose = False)
        if (year_str > -1):
            self.data = self.data[self.data['year'] >= str(year_str)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (year_end > -1):
            self.data = self.data[self.data['year'] <= str(year_end)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(sources) > 0):
            src_idx = []
            for source in sources:
                for i in range(0, len(self.jou)):
                    if (source == self.jou[i][0]):
                        src_idx.append(i)
            if (len(src_idx) > 0):
                self.data = self.data.iloc[src_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (core == 1 or core == 2 or core == 3 or core == 12 or core == 23):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            c2    = int(value[-1]*(2/3))
            if (core ==  1):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c1]
            if (core ==  2):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1 and value[i] <= c2]
            if (core ==  3):
               key = [key[i] for i in range(0, len(key)) if value[i] > c2]
            if (core == 12):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c2]
            if (core == 23):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1]
            sources   = self.data['abbrev_source_title'].str.lower()
            self.data = self.data[sources.isin(key)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(country) > 0):
            ctr_idx   = [i for i in range(0, len(self.ctr)) if any(x in country for x in self.ctr[i])] 
            if (len(ctr_idx) > 0):
                self.data = self.data.iloc[ctr_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (len(language) > 0):
            self.data = self.data[self.data['language'].isin(language)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (abstract == True):
            self.data = self.data[self.data['abstract'] != 'UNKNOW']
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        self.__update_vb()
        self.__make_bib(verbose = True)
        return
    
    #from functools import lru_cache
    #def __lev_dist(a, b): 
        #@lru_cache(None)
        #def min_dist(s1, s2):
            #if s1 == len(a) or s2 == len(b):
                #return len(a) - s1 + len(b) - s2
            #if a[s1] == b[s2]:
                #return min_dist(s1 + 1, s2 + 1)
            #return 1 + min(min_dist(s1, s2 + 1),      # insert character
                           #min_dist(s1 + 1, s2),      # delete character
                           #min_dist(s1 + 1, s2 + 1),  # replace character
                           #)
        #return min_dist(0, 0)
        
    # Function: Get Duplicates Index
    def find_duplicates(self, u_list):
        duplicates = []
        indices    = []
        for i in range(0, len(u_list)):
            x = u_list[i]
            if (u_list.count(x) > 1 and x not in indices):
                u_list.index(u_list[i])
                duplicates.append([i for i in range(0, len(u_list)) if u_list[i] == x])
                indices.append(x)
        return indices, duplicates
    
    # Function: Fuzzy String Matcher
    def fuzzy_matcher(self, entry = 'aut', cut_ratio = 0.80, verbose = True): # 'aut', 'inst'
        if (entry == 'aut'):
            u_lst = [item for item in self.u_aut]
        elif (entry == 'inst'):
            u_lst = [item for item in self.u_uni]
        else:
            u_lst = [item for item in entry]
        fuzzy_lst = [[] for item in u_lst ]
        idx       = [i for i in range(0, len(u_lst))]
        i         = 0
        while (len(idx) != 0):
            if (i in idx):
                idx.remove(i)
            for j in idx:
                ratio = SequenceMatcher(None, u_lst[i], u_lst[j]).ratio()
                if (ratio >= cut_ratio and ratio < 1):
                    fuzzy_lst[i].append(u_lst[j])
                    if (j in idx):
                        idx.remove(j)
            i = i + 1
        indices = [i for i, x in enumerate(fuzzy_lst) if x == []]
        for i in sorted(indices, reverse = True):
            del fuzzy_lst[i]
            del u_lst[i]
        fuzzy_dict = dict(zip(u_lst, fuzzy_lst))
        if (verbose == True):
            for key, value in fuzzy_dict.items():
               print(str(key)+': '+str('; '.join(value)))
        return fuzzy_dict

    # Function: Merge Datatbase
    def merge_database(self, file_bib, db, del_duplicated):
        old_vb   = [item for item in self.vb]
        old_size = self.data.shape[0]
        print('############################################################################')
        print('')
        print('Original Database')
        print('')
        for i in range(0, len(old_vb)):
            print(old_vb[i])
        print('')
        print('############################################################################')
        print('')
        print('Added Database')
        print('')
        data, _    = self.__read_bib(file_bib, db, del_duplicated)
        self.data  = pd.concat([self.data, data]) 
        self.data  = self.data.reset_index(drop = True)
        self.data  = self.data.fillna('UNKNOW')
        duplicated = self.data['doi'].duplicated() # self.data = self.data.drop_duplicates(subset = 'doi', keep = 'first')
        title      = self.data['title']
        title      = title.to_list()
        title      = self.clear_text(title, stop_words  = [], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [])
        t_dupl     = pd.Series(title).duplicated()
        for i in range(0, duplicated.shape[0]):
            if (self.data.loc[i, 'doi'] == 'UNKNOW'):
                duplicated[i] = False
            if (t_dupl[i] == True):
                duplicated[i] = True
        idx        = list(duplicated.index[duplicated])
        self.data.drop(idx, axis = 0, inplace = True)
        self.data  = self.data.reset_index(drop = True)
        size       = self.data.shape[0]
        self.__make_bib(verbose = True)
        dt         = self.data['document_type'].value_counts()
        dt         = dt.sort_index(axis = 0)
        self.vb    = []
        print('')
        print('############################################################################')
        print('')
        print('Merging Information:')
        print('')
        print( 'A Total of ' + str(size) + ' Documents were Found ( ' + str(size - old_size) + ' New Documents from the Added Database )')
        self.vb.append('A Total of ' + str(size) + ' Documents were Found')
        print('')
        for i in range(0, dt.shape[0]):
            print(dt.index[i], ' = ', dt[i])
            self.vb.append(dt.index[i] + ' = ' + str(dt[i]))
        print('')
        print('############################################################################')
        return
    
    # Function: Merge Author
    def merge_author(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'author'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'author'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Institution
    def merge_institution(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'affiliation'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'affiliation'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Country
    def merge_country(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'affiliation'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'affiliation'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Language
    def merge_language(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'language'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'language'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return
    
    # Function: Merge Source
    def merge_source(self, get = [], replace_for = 'name'):
        for name in get:
            for i in range(0, self.data.shape[0]):
                target = self.data.loc[i, 'abbrev_source_title'].lower()
                if (name.lower() in target):
                    self.data.loc[i, 'abbrev_source_title'] = target.replace(name, replace_for)
        self.__make_bib(verbose = False)
        return

    # Function: Transform Hex to RGBa
    def __hex_rgba(self, hxc = '#ba6200', alpha = 0.15):
        if (hxc.find('#') == 0):
            hxc  = hxc.lstrip('#')
            rgb  = tuple(int(hxc[i:i+2], 16) for i in (0, 2, 4))
            rgba = 'rgba('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+','+str(alpha)+')'
        else:
            rgba = 'black'
        return rgba
    
    #############################################################################
    
    # Function: EDA .bib docs
    def eda_bib(self):
        report = []
        report.append(['Timespan', str(self.date_str)+'-'+str(self.date_end)])
        report.append(['Total Number of Countries', len(self.u_ctr)])
        report.append(['Total Number of Institutions', len(self.u_uni)])
        report.append(['Total Number of Sources', len(self.u_jou)])
        report.append(['Total Number of References', len(self.u_ref)])
        report.append(['Total Number of Languages', len(self.u_lan)])
        for i in range(0, len(self.u_lan)):
            report.append(['--'+self.u_lan[i]+' (# of docs)', self.lan_count[i]])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Documents', self.data.shape[0]])
        for i in range(0, self.doc_types.shape[0]):
            report.append(['--'+self.doc_types.index[i], self.doc_types[i]])
        report.append(['Average Documents per Author',self. av_doc_aut])
        report.append(['Average Documents per Institution', round(sum(self.uni_count)/len(self.uni_count), 2)])
        report.append(['Average Documents per Source', round(sum(self.jou_count)/len(self.jou_count), 2)])
        report.append(['Average Documents per Year', self.av_d_year])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Authors', len(self.u_aut)])
        report.append(['Total Number of Authors Keywords', len(self.u_auk)])
        report.append(['Total Number of Authors Keywords Plus', len(self.u_kid)])
        report.append(['Total Single-Authored Documents', self.aut_single])
        report.append(['Total Multi-Authored Documents', len(self.aut_multi)])
        report.append(['Average Collaboration Index', self.dy_c_year.iloc[-1, -1]])
        report.append(['Max H-Index', max(self.aut_h)])
        report.append(['-//-', '-//-'])
        report.append(['Total Number of Citations', sum(self.citation)])
        report.append(['Average Citations per Author', round(sum(self.citation)/len(self.u_aut), 2)])
        report.append(['Average Citations per Institution', round(sum(self.citation)/len(self.u_uni), 2)])
        report.append(['Average Citations per Document', self.av_c_doc])
        report.append(['Average Citations per Source', round(sum(self.jou_cit)/len(self.jou_cit), 2)])
        report.append(['-//-', '-//-'])
        report_df = pd.DataFrame(report, columns = ['Main Information', 'Results'])
        return report_df

    ##############################################################################

    # Function: Read .bib File
    def __read_bib(self, bib, db = 'scopus', del_duplicated = True):
        self.vb = []
        db      = db.lower()
        f_file  = open(bib, 'r', encoding = 'utf8')
        f_lines = f_file.read()
        f_list  = f_lines.split('\n')
        if (db == 'wos'):
            f_list_ = []
            for i in range(0, len(f_list)):
                if (f_list[i][:3] != '   '):
                    f_list_.append(f_list[i])
                else:
                    if (f_list_[-1].find('Cited-References') != -1):
                        f_list[i] = f_list[i].replace(';', ',')
                    if (f_list_[-1].find('Cited-References') == -1):
                        f_list_[-1] = f_list_[-1] + f_list[i]
                    else:
                        f_list[i]   = f_list[i].replace(';', ',')
                        f_list_[-1] = f_list_[-1] + ';' + f_list[i]
            f_list = f_list_
        if (db == 'pubmed'):
            f_list_ = []
            for i in range(0, len(f_list)):
                if (i == 0 and f_list[i][:6] != '      ' ):
                    f_list_.append(f_list[i])
                elif (i > 0 and f_list[i][:6] != '      ' and f_list[i][:6] != f_list[i-1][:6] and (f_list[i][:6].lower() != 'fau - ' and f_list[i][:6].lower() != 'au  - ' and f_list[i][:6].lower() != 'auid- ' and f_list[i][:6].lower() != 'ad  - ')):
                    f_list_.append(f_list[i])
                elif (i > 0 and f_list[i][:6] != '      ' and f_list[i][:6] == f_list[i-1][:6] and (f_list[i][:6].lower() != 'fau - ' and f_list[i][:6].lower() != 'au  - ' and f_list[i][:6].lower() != 'auid- ' and f_list[i][:6].lower() != 'ad  - ' and f_list[i][:6].lower() != 'pt  - ')):
                    f_list_[-1] = f_list_[-1] + '; ' + f_list[i][6:]
                elif (f_list[i][:6] == '      '):
                    f_list_[-1] = f_list_[-1] + f_list[i][6:]
                elif (f_list[i][:6] == 'FAU - '):
                    f_list_.append(f_list[i])
                    j = i + 1
                    while (len(f_list[j]) != 0):
                        j = j + 1
                        if (f_list[j][:6].lower() == 'fau - '):
                            f_list_[-1] = f_list_[-1] + '; ' + f_list[j][6:]
                            f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                elif (f_list[i][:6] == 'AU  - '):
                    f_list_.append(f_list[i])
                    j = i + 1
                    while (len(f_list[j]) != 0):
                        j = j + 1
                        if (f_list[j][:6].lower() == 'au  - '):
                            f_list_[-1] = f_list_[-1] + ' and ' + f_list[j][6:]
                            f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                elif (f_list[i][:6] == 'AUID- '):
                    f_list_.append(f_list[i])
                    j = i + 1
                    while (len(f_list[j]) != 0):
                        j = j + 1
                        if (f_list[j][:6].lower() == 'auid- '):
                            f_list_[-1] = f_list_[-1] + '; ' + f_list[j][6:]
                            f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                elif (f_list[i][:6] == 'AD  - '):
                    f_list_.append(f_list[i])
                    j = i + 1
                    while (len(f_list[j]) != 0):
                        j = j + 1
                        if (f_list[j][:6].lower() == 'ad  - '):
                            f_list_[-1] = f_list_[-1] + f_list[j][6:]
                            f_list[j]   = f_list[j][:6].lower() + f_list[j][6:]
                elif (f_list[i][:6] == 'PT  - '):
                    f_list_.append(f_list[i])
                    j = i + 1
                    while (len(f_list[j]) != 0):
                        j = j + 1
                        if (f_list[j][:6].lower() == 'pt  - '):
                            f_list[j] = f_list[j][:6].lower() + f_list[j][6:]
            f_list = [item for item in f_list_]
            for i in range(0, len(f_list)):
                if (len(f_list[i]) > 0):
                    if (f_list[i][4] == '-'):
                        f_list[i] = f_list[i][:4] + '=' + f_list[i][5:]
                    if (f_list[i][:3] == 'LID'):
                        f_list[i] = f_list[i].replace(' [doi]', '')
                        #f_list[i] = f_list[i].replace(' [pii]', '')
                        #f_list[i] = f_list[i].replace(' [isbn]', '')
                        #f_list[i] = f_list[i].replace(' [ed]', '')
                        #f_list[i] = f_list[i].replace(' [editor]', '')
                        #f_list[i] = f_list[i].replace(' [book]', '')
                        #f_list[i] = f_list[i].replace(' [bookaccession]', '')
        lhs = []
        rhs = []
        doc = 0
        for i in range(0, len(f_list)):
          if (f_list[i].find('@') == 0 or f_list[i][:4].lower() == 'pmid'):  
            lhs.append('doc_start')
            rhs.append('doc_start')
            if (db == 'pubmed'):
                lhs.append('note')
                rhs.append('0')
                lhs.append('source')
                rhs.append('PubMed')
            if (db == 'wos'):
                lhs.append('source')
                rhs.append('WoS')
            doc = doc + 1
          if (f_list[i].find('=') != -1 and f_list[i].find(' ') != 0):
            lhs.append(f_list[i].split('=')[0].lower().strip())
            rhs.append(f_list[i].split('=')[1].replace('{', '').replace('},', '').replace('}', '').replace('}},', '').strip())
          elif (f_list[i].find(' ') == 0 and i!= 0 and rhs[-1] != 'doc_start'):
            rhs[-1] = rhs[-1]+' '+f_list[i].replace('{', '').replace('},', '').replace('}', '').replace('}},', '').strip()
        if (db == 'wos'):
            for i in range(0, len(lhs)):
                if (lhs[i] == 'affiliation'):
                    lhs[i] = 'affiliation_'
                if (lhs[i] == 'affiliations'):
                    lhs[i] = 'affiliation'
                if (lhs[i] == 'article-number'):
                    lhs[i] = 'art_number'
                if (lhs[i] == 'cited-references'):
                    lhs[i] = 'references'
                if (lhs[i] == 'keywords'):
                    lhs[i] = 'author_keywords'
                if (lhs[i] == 'journal-iso'):
                    lhs[i] = 'abbrev_source_title'
                if (lhs[i] == 'keywords-plus'):
                    lhs[i] = 'keywords'
                if (lhs[i] == 'note'):
                    lhs[i] = 'note_'
                if (lhs[i] == 'times-cited'):
                    lhs[i] = 'note'
                if (lhs[i] == 'type'):
                    lhs[i] = 'document_type'
                lhs[i] = lhs[i].replace('-', '_')
        if (db == 'pubmed'):
            for i in range(0, len(lhs)):
                if (lhs[i] == 'ab'):
                    lhs[i] = 'abstract'
                if (lhs[i] == 'ad'):
                    lhs[i] = 'affiliation'
                if (lhs[i] == 'au'):
                    lhs[i] = 'author'
                if (lhs[i] == 'auid'):
                    lhs[i] = 'orcid'
                if (lhs[i] == 'fau'):
                    lhs[i] = 'full_author'
                if (lhs[i] == 'lid'):
                    lhs[i] = 'doi'
                if (lhs[i] == 'dp'):
                    lhs[i] = 'year'
                    rhs[i] = rhs[i][:4]
                if (lhs[i] == 'ed'):
                    lhs[i] = 'editor'
                if (lhs[i] == 'ip'):
                    lhs[i] = 'issue'
                if (lhs[i] == 'is'):
                    lhs[i] = 'issn'
                if (lhs[i] == 'isbn'):
                    lhs[i] = 'isbn'
                if (lhs[i] == 'jt'):
                    lhs[i] = 'journal'
                if (lhs[i] == 'la'):
                    lhs[i] = 'language'
                    if (rhs[i] in self.language_names.keys()):
                        rhs[i] = self.language_names[rhs[i]]
                if (lhs[i] == 'mh'):
                    lhs[i] = 'keywords'
                if (lhs[i] == 'ot'):
                    lhs[i] = 'author_keywords'
                if (lhs[i] == 'pg'):
                    lhs[i] = 'pages'
                if (lhs[i] == 'pt'):
                    lhs[i] = 'document_type'
                if (lhs[i] == 'pmid'):
                    lhs[i] = 'pubmed_id'
                if (lhs[i] == 'ta'):
                    lhs[i] = 'abbrev_source_title'
                if (lhs[i] == 'ti'):
                    lhs[i] = 'title'
                if (lhs[i] == 'vi'):
                    lhs[i] = 'volume'
        labels       = list(set(lhs))
        labels.remove('doc_start')
        sanity_check = ['abbrev_source_title', 'abstract', 'address', 'affiliation', 'art_number', 'author', 'author_keywords', 'chemicals_cas', 'coden', 'correspondence_address1', 'document_type', 'doi', 'editor', 'funding_details', 'funding_text\xa01', 'funding_text\xa02', 'funding_text\xa03', 'isbn', 'issn', 'journal', 'keywords', 'language', 'note', 'number', 'page_count', 'pages', 'publisher', 'pubmed_id', 'references', 'source', 'sponsors', 'title', 'tradenames', 'url', 'volume', 'year']
        for item in sanity_check:
            if (item not in labels):
                labels.append(item)
        labels.sort()      
        values      = [i for i in range(0, len(labels))] 
        labels_dict = dict(zip(labels, values))
        data        = pd.DataFrame(index = range(0, doc), columns = labels)
        count       = -1
        for i in range(0, len(rhs)):
          if (lhs[i] == 'doc_start'):
            count = count + 1
          else:
            data.iloc[count, labels_dict[lhs[i]]] = rhs[i]
        entries = list(data.columns)
        
        # WoS -> Scopus
        data['document_type'] = data['document_type'].replace('Article; Early Access','Article in Press')
        data['document_type'] = data['document_type'].replace('Article; Proceedings Paper','Proceedings Paper')
        data['document_type'] = data['document_type'].replace('Article; Proceedings Paper','Proceedings Paper')
        data['document_type'] = data['document_type'].replace('Article; Discussion','Article')
        data['document_type'] = data['document_type'].replace('Article; Letter','Article')
        data['document_type'] = data['document_type'].replace('Article; Excerpt','Article')
        data['document_type'] = data['document_type'].replace('Article; Chronology','Article')
        data['document_type'] = data['document_type'].replace('Article; Correction','Article')
        data['document_type'] = data['document_type'].replace('Article; Correction, Addition','Article')
        data['document_type'] = data['document_type'].replace('Article; Data Paper','Article')
        data['document_type'] = data['document_type'].replace('Art Exhibit Review','Review')
        data['document_type'] = data['document_type'].replace('Dance Performance Review','Review')
        data['document_type'] = data['document_type'].replace('Music Performance Review','Review')
        data['document_type'] = data['document_type'].replace('Music Score Review','Review')
        data['document_type'] = data['document_type'].replace('Film Review','Review')
        data['document_type'] = data['document_type'].replace('TV Review, Radio Review','Review')
        data['document_type'] = data['document_type'].replace('TV Review, Radio Review, Video','Review')
        data['document_type'] = data['document_type'].replace('Theater Review, Video','Review')
        data['document_type'] = data['document_type'].replace('Database Review','Review')
        data['document_type'] = data['document_type'].replace('Record Review','Review')
        data['document_type'] = data['document_type'].replace('Software Review','Review')
        data['document_type'] = data['document_type'].replace('Hardware Review','Review')
        
        # PubMed -> Scopus
        data['document_type'] = data['document_type'].replace('Clinical Study','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial Protocol','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase I','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase II','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase III','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Phase IV','Article')
        data['document_type'] = data['document_type'].replace('Clinical Trial, Veterinary','Article')
        data['document_type'] = data['document_type'].replace('Comparative Study','Article')
        data['document_type'] = data['document_type'].replace('Controlled Clinical Trial','Article')
        data['document_type'] = data['document_type'].replace('Corrected and Republished Article','Article')
        data['document_type'] = data['document_type'].replace('Duplicate Publication','Article')
        data['document_type'] = data['document_type'].replace('Essay','Article')
        data['document_type'] = data['document_type'].replace('Historical Article','Article')
        data['document_type'] = data['document_type'].replace('Journal Article','Article')
        data['document_type'] = data['document_type'].replace('Letter','Article')
        data['document_type'] = data['document_type'].replace('Meta-Analysis','Article')
        data['document_type'] = data['document_type'].replace('Randomized Controlled Trial','Article')
        data['document_type'] = data['document_type'].replace('Randomized Controlled Trial, Veterinary','Article')
        data['document_type'] = data['document_type'].replace('Research Support, N.I.H., Extramural','Article')
        data['document_type'] = data['document_type'].replace('Research Support, N.I.H., Intramural','Article')
        data['document_type'] = data['document_type'].replace("Research Support, Non-U.S. Gov't",'Article')
        data['document_type'] = data['document_type'].replace("Research Support, U.S. Gov't, Non-P.H.S.",'Article')
        data['document_type'] = data['document_type'].replace("Research Support, U.S. Gov't, P.H.S.",'Article')
        data['document_type'] = data['document_type'].replace('Research Support, U.S. Government','Article')
        data['document_type'] = data['document_type'].replace('Research Support, American Recovery and Reinvestment Act','Article')
        data['document_type'] = data['document_type'].replace('Technical Report','Article')
        data['document_type'] = data['document_type'].replace('Twin Study','Article')
        data['document_type'] = data['document_type'].replace('Validation Study','Article')
        data['document_type'] = data['document_type'].replace('Clinical Conference','Conference Paper')
        data['document_type'] = data['document_type'].replace('Congress','Conference Paper')
        data['document_type'] = data['document_type'].replace('Consensus Development Conference','Conference Paper')
        data['document_type'] = data['document_type'].replace('Consensus Development Conference, NIH','Conference Paper')
        data['document_type'] = data['document_type'].replace('Systematic Review','Review')
        data['document_type'] = data['document_type'].replace('Scientific Integrity Review','Review')
        
        if (del_duplicated == True and 'doi' in entries):
            duplicated = data['doi'].duplicated()
            title      = data['title']
            title      = title.to_list()
            title      = self.clear_text(title, stop_words  = [], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [])
            t_dupl     = pd.Series(title).duplicated()
            for i in range(0, duplicated.shape[0]):
                if (data.loc[i, 'doi'] == 'UNKNOW' or pd.isnull(data.loc[i, 'doi'])):
                    duplicated[i] = False
                if (t_dupl[i] == True):
                    duplicated[i] = True
            idx        = list(duplicated.index[duplicated])
            data.drop(idx, axis = 0, inplace = True)
            data       = data.reset_index(drop = True)
            string_vb  = 'A Total of ' + str(doc-len(idx)) + ' Documents were Found ( ' + str(doc) + ' Documents and '+ str(len(idx)) + ' Duplicates )'
            self.vb.append(string_vb)
        else:
            string_vb  = 'A Total of ' + str(doc) + ' Documents were Found' 
            self.vb.append(string_vb)
        if (db == 'wos' and 'type' in entries):
            data['document_type'] = data['type']
        if ('document_type' in entries):
            types     = list(data['document_type'].replace(np.nan, 'UNKNOW'))
            u_types   = list(set(types))
            u_types.sort()
            string_vb = ''
            self.vb.append(string_vb)
            for tp in u_types:
                string_vb = tp + ' = ' + str(types.count(tp))
                self.vb.append(string_vb)
        data.fillna('UNKNOW', inplace = True)
        data['keywords']        = data['keywords'].apply(lambda x: x.replace(',',';'))
        data['author_keywords'] = data['author_keywords'].apply(lambda x: x.replace(',',';'))
        if (db == 'wos'):
            idx     = data[data['year'] == 'UNKNOW'].index.values
            idx_val = [data.loc[i, 'da'][:4] for i in idx]
            for i in range(0, len(idx)):
                data.iloc[idx[i], -1] = idx_val[i]
        data = data.reindex(sorted(data.columns), axis = 1)
        return data, entries
    
    # Function: Update Verbose
    def __update_vb(self):
        self.vb   = []
        self.vb.append('A Total of ' + str(self.data.shape[0]) + ' Documents Remains' )
        types     = list(self.data['document_type'])
        u_types   = list(set(types))
        u_types.sort()
        string_vb = ''
        self.vb.append(string_vb)
        for tp in u_types:
            string_vb = tp + ' = ' + str(types.count(tp))
            self.vb.append(string_vb)
        return
    ##############################################################################
    
    # Function: Get Entries
    def __get_str(self, entry = 'references', s = ';', lower = True, sorting = True):
        info = []
        for i in range(0, self.data[entry].shape[0]):
            e = self.data[entry][i]
            if (isinstance(e, str) == True):
                strg = e.split(s)
                strg = [item.strip() for item in strg if item.strip() != 'note']
                strg = [' '.join(item.split()) for item in strg]
                if (lower == True):
                    strg = [item.lower() for item in strg]
                info.append(strg)
            else:
                info.append([])
        unique_info = [item for sublist in info for item in sublist]
        unique_info = list(set(unique_info))
        if (unique_info[0] == ''):
            unique_info = unique_info[1:]
        if (sorting == True):
            unique_info.sort()
        return info, unique_info
    
    # Function: Get Citations
    def __get_citations(self, series):
        citation = [item.lower().replace('cited by ' , '') for item in list(series)] 
        citation = [item.lower().replace('cited by: ', '') for item in list(citation)] 
        for i in range(0, len(citation)):
            idx = citation[i].find(';')
            if (idx >= 0):
                try:
                    citation[i] = int(citation[i][:idx])
                except:
                    citation[i] = int(re.search(r'\d+', citation[i]).group())
            else:
                try:
                    citation[i] = int(citation[i])
                except:
                    citation[i] = int(re.search(r'\d+', citation[i]).group()) 
        return citation
    
    # Function: Get Past Citations per Year
    def __get_past_citations_year(self):
        df      = self.data[['author', 'title', 'doi', 'year', 'references']]
        df      = df.sort_values(by = ['year'])
        df      = df.reset_index(drop = True)
        c_count = [0]*df.shape[0]
        c_year  = list(df['year'])
        for i in range(0, df.shape[0]):
            title = df.iloc[i, 1]
            for j in range(i, len(self.ref)):
                for k in range(0, len(self.ref[j])):
                    if (title.lower() in self.ref[j][k].lower()):
                        c_count[j] = c_count[j] + 1
        c_year_  = list(set(c_year))
        c_year_.sort()
        c_count_ = [0]*len(c_year_)
        for i in range(0, len(c_year)):
            c_count_[c_year_.index(c_year[i])] = c_count_[c_year_.index(c_year[i])] + c_count[i]
        return c_year_, c_count_
    
    # Function: Get Countries
    def __get_countries(self):
        if   (self.data_base == 'scopus' or self.data_base == 'pubmed'):
            df = self.data['affiliation']
            df = df.str.lower()
        elif (self.data_base == 'wos'):
            df = self.data['affiliation_']
            df = df.str.replace(' USA',            ' United States of America',   case = False, regex = True)
            df = df.str.replace('ENGLAND',         'United Kingdom',              case = False, regex = True)
            df = df.str.replace('Antigua & Barbu', 'Antigua and Barbuda',         case = False, regex = True)
            df = df.str.replace('Bosnia & Herceg', 'Bosnia and Herzegovina',      case = False, regex = True)
            df = df.str.replace('Cent Afr Republ', 'Central African Republic',    case = False, regex = True)
            df = df.str.replace('Dominican Rep',   'Dominican Republic',          case = False, regex = True)
            df = df.str.replace('Equat Guinea',    'Equatorial Guinea',           case = False, regex = True)
            df = df.str.replace('Fr Austr Lands',  'French Southern Territories', case = False, regex = True)
            df = df.str.replace('Fr Polynesia',    'French Polynesia',            case = False, regex = True)
            df = df.str.replace('Malagasy Republ', 'Madagascar',                  case = False, regex = True)
            df = df.str.replace('Mongol Peo Rep',  'Mongolia',                    case = False, regex = True)
            df = df.str.replace('Neth Antilles',   'Saint Martin',                case = False, regex = True)
            df = df.str.replace('North Ireland',   'Ireland',                     case = False, regex = True)
            df = df.str.replace('Peoples R China', 'China',                       case = False, regex = True)
            df = df.str.replace('Rep of Georgia',  'Georgia',                     case = False, regex = True)
            df = df.str.replace('Sao Tome E Prin', 'Sao Tome and Principe',       case = False, regex = True)
            df = df.str.replace('St Kitts & Nevi', 'Saint Kitts and Nevis',       case = False, regex = True)
            df = df.str.replace('Trinid & Tobago', 'Trinidad and Tobago',         case = False, regex = True)
            df = df.str.replace('U Arab Emirates', 'United Arab Emirates',        case = False, regex = True)
            df = df.str.lower()
            for i in range(0, len(self.aut)):
                for j in range(0, len(self.aut[i])):
                    df = df.str.replace(self.aut[i][j], self.aut[i][j].replace('.', ''), regex = True)
        ctrs = [[] for i in range(0, df.shape[0])]
        if (self.data_base == 'scopus'):
            for i in range(0, df.shape[0]):
                affiliations = str(df[i]).strip().split(';')
                for affiliation in affiliations:
                    for country in self.country_names:
                        if (country.lower() in affiliation.lower()):
                            ctrs[i].append(country)
                            break
        if (self.data_base == 'pubmed'):
            for i in range(0, df.shape[0]):
                affiliations = str(df[i]).strip().split(',')
                for affiliation in affiliations:
                    for country in self.country_names:
                        if (country.lower() in affiliation.lower()):
                            ctrs[i].append(country)
                            break
        elif (self.data_base == 'wos'):
           for i in range(0, df.shape[0]): 
               affiliations = str(df[i]).strip().split('.')[:-1]
               for affiliation in affiliations:
                    for j in range(0, len(self.aut[i])):
                        for country in self.country_names:
                            if (country.lower() in affiliation.lower() and self.aut[i][j].lower().replace('.', '') in affiliation.lower()):
                                ctrs[i].append(country)
                                break
        for i in range(0, len(ctrs)):
            while len(self.aut[i]) > len(ctrs[i]):
                if (len(ctrs[i]) == 0):
                    ctrs[i].append('UNKNOW')
                ctrs[i].append(ctrs[i][-1])
            if (len(ctrs[i]) == 0):
                ctrs[i].append('UNKNOW')
        u_ctrs = [item for sublist in ctrs for item in sublist]
        u_ctrs = list(set(u_ctrs))
        if (len(u_ctrs[0]) == 0):
            u_ctrs = u_ctrs[1:]
        return ctrs, u_ctrs
  
    # Function: Get Institutions
    def __get_institutions(self):
        if   (self.data_base == 'scopus' or self.data_base == 'pubmed'):
            df = self.data['affiliation']
            df = df.str.lower()
        elif (self.data_base == 'wos'):
            df = self.data['affiliation_']
            df = df.str.lower()
            for i in range(0, len(self.aut)):
                for j in range(0, len(self.aut[i])):
                    df = df.str.replace(self.aut[i][j], self.aut[i][j].replace('.', ''), regex = True)
        inst  = [[] for i in range(0, df.shape[0])]
        inst_ = [[] for i in range(0, df.shape[0])]
        if  (self.data_base == 'scopus'):
            for i in range(0, df.shape[0]):
                affiliations = str(df[i]).split(';')
                for affiliation in affiliations:
                    for institution in self.institution_names:
                        if (institution.lower() in affiliation.lower()):
                            if (affiliation.strip() not in inst[i]):
                                inst[i].append(affiliation.strip())
                            break
            for i in range(0, len(inst)):
                for j in range(0, len(inst[i])):
                    item = inst[i][j].split(',')
                    for institution in self.institution_names:
                        idx = [k for k in range(0, len(item)) if institution in item[k].lower()]
                        if (len(idx) > 0):
                            institution_name = item[idx[0]]
                            institution_name = ' '.join(institution_name.split())
                            inst_[i].append(institution_name)
                            break
        if  (self.data_base == 'pubmed'):
            for i in range(0, df.shape[0]):
                affiliations = str(df[i]).split(',')
                for affiliation in affiliations:
                    for institution in self.institution_names:
                        if (institution.lower() in affiliation.lower()):
                            if (affiliation.strip() not in inst[i]):
                                inst[i].append(affiliation.strip())
                            break
            for i in range(0, len(inst)):
                for j in range(0, len(inst[i])):
                    item = inst[i][j].split(',')
                    for institution in self.institution_names:
                        idx = [k for k in range(0, len(item)) if institution in item[k].lower()]
                        if (len(idx) > 0):
                            institution_name = item[idx[0]]
                            institution_name = ' '.join(institution_name.split())
                            inst_[i].append(institution_name)
                            break
        elif (self.data_base == 'wos'):
            for i in range(0, df.shape[0]): 
               affiliations = str(df[i]).strip().split('.')[:-1]
               for affiliation in affiliations:
                    for j in range(0, len(self.aut[i])):
                        for institution in self.institution_names:
                            if (institution.lower() in affiliation.lower() and self.aut[i][j].lower().replace('.', '') in affiliation.lower()):
                                if (affiliation.strip() not in inst[i]):
                                    inst[i].append(affiliation.strip().replace('\&', 'and'))
                                break            
            for i in range(0, len(inst)):
                for j in range(0, len(inst[i])):
                    item = inst[i][j].split(',')
                    for institution in self.institution_names:
                        idx = [k for k in range(0, len(item)) if institution in item[k].lower()]
                        if (len(idx) > 0):
                            institution_name = item[idx[0]]
                            institution_name = ' '.join(institution_name.split())
                            inst_[i].append(institution_name)
                            break
        for i in range(0, len(inst_)):
            while len(self.aut[i]) > len(inst_[i]):
                if (len(inst_[i]) == 0):
                    inst_[i].append('UNKNOW')
                inst_[i].append(inst_[i][-1])
            if (len(inst_[i]) == 0):
                inst_[i].append('UNKNOW')
        u_inst = [item for sublist in inst_ for item in sublist]
        u_inst = list(set(u_inst))
        if (len(u_inst[0]) == 0):
            u_inst = u_inst[1:]
        return inst_, u_inst
    
    # Function: Get Counts
    def __get_counts(self, u_ent, ent, acc = []):
        counts = []
        for i in range(0, len(u_ent)):
            ents = 0
            for j in range(0, len(ent)):
                if (u_ent[i] in ent[j] and len(acc) == 0):
                    ents = ents + 1
                elif (u_ent[i] in ent[j] and len(acc) > 0):
                    ents = ents + acc[j]
            counts.append(ents)
        return counts
    
    # Function: Get Count Year
    def __get_counts_year(self, u_ent, ent):
        years = list(range(self.date_str, self.date_end+1))
        df_counts = pd.DataFrame(np.zeros((len(u_ent),len(years))))
        for i in range(0, len(u_ent)):
            for j in range(0, len(ent)):
                if (u_ent[i] in ent[j]):
                    k = years.index(int(self.dy[j]))
                    df_counts.iloc[i, k] = df_counts.iloc[i, k] + 1
        return df_counts
    
    # Function: Get Collaboration Year
    def __get_collaboration_year(self):
        max_aut         = list(set([str(item) for item in self.aut_docs]))
        max_aut         = sorted(max_aut, key = self.natsort)
        n_collaborators = ['n = ' + i for i in max_aut]
        n_collaborators.append('ci')
        years           = list(range(self.date_str, self.date_end+1))
        years           = [str(int(item)) for item in years]
        years.append('Total')
        dy_collab_year = pd.DataFrame(np.zeros((len(years), len( n_collaborators))), index = years, columns = n_collaborators)
        for k in range(0, len(self.aut)):
            i                        = str(int(self.dy[k]))
            j                        = ['n = ' + str(len(self.aut[k]))]
            dy_collab_year.loc[i, j] = dy_collab_year.loc[i, j] + 1    
        dy_collab_year.iloc[-1, :] = dy_collab_year.sum(axis = 0)
        dy_collab_year.iloc[:, -1] = dy_collab_year.sum(axis = 1)
        for i in range(0, dy_collab_year.shape[0]):
            ci                         = sum([ (j+1) * dy_collab_year.iloc[i, j] for j in range(0, dy_collab_year.shape[1]-1)])
            if (dy_collab_year.iloc[i, -1] > 0):
                dy_collab_year.iloc[i, -1] = round(ci / dy_collab_year.iloc[i, -1], 2)               
        return dy_collab_year
    
    # Function: Get Reference Year
    def __get_ref_year(self):
        dy_ref = []
        for item in self.u_ref:
            years = ['-2']
            while years[-1] != '-1':
                a1, a2, a3 = '1', '8', '0' 
                b1, b2, b3 = str(self.date_end)[0], str(self.date_end)[1], str(self.date_end)[2]
                match_1    = re.match(r'.*(['+a1+'-'+a1+']['+a2+'-9]['+a3+'-9][0-9])', item)
                match_2    = re.match(r'.*(['+b1+'-'+b1+']['+b2+'-'+b2+'][0-'+b3+'][0-9])', item)
                if (match_1 is not None and match_2 is not None and int(match_1.group(1)) >= int(match_2.group(1)) ):
                    years.append(match_1.group(1))
                    item = item.replace(match_1.group(1), '')
                elif (match_1 is not None and match_2 is not None and int(match_1.group(1)) < int(match_2.group(1)) ):
                    if (int(match_2.group(1)) <= self.date_end):
                        years.append(match_2.group(1))
                    item = item.replace(match_2.group(1), '')
                elif (match_1 is not None):
                    years.append(match_1.group(1))
                    item = item.replace(match_1.group(1), '')
                elif (match_2 is not None):
                    if (int(match_2.group(1)) <= self.date_end):
                        years.append(match_2.group(1))
                    item = item.replace(match_2.group(1), '')
                else:
                    years.append('-1')
            years = [int(year) for year in years]
            dy_ref.append(max(years))
        return dy_ref
    
    ##############################################################################
    
    # Function: Wordcloud 
    def word_cloud_plot(self, entry = 'kwp', size_x = 10, size_y = 10, wordsn = 500):
        if  (entry == 'kwp'):
            corpora = ' '.join(self.kid_)
        elif (entry == 'kwa'):
            corpora = ' '.join(self.auk_)  
        elif (entry == 'abs'):
            abs_    = self.data['abstract']
            abs_    = list(abs_)
            abs_    = [x for x in abs_ if str(x) != 'nan']
            corpora = ' '.join(abs_)
        elif (entry == 'title'):
            tit_    = self.data['title']
            tit_    = list(tit_)
            tit_    = [x for x in tit_ if str(x) != 'nan']
            corpora = ' '.join(tit_)
        wordcloud = WordCloud(background_color = 'white', 
                              max_words        = wordsn, 
                              contour_width    = 25, 
                              contour_color    = 'steelblue', 
                              collocations     = False, 
                              width            = 1600, 
                              height           = 800
                              )
        wordcloud.generate(corpora)
        plt.figure(figsize = (size_x, size_y), facecolor = 'k')
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.show()
        return
    
    # Function: Get Top N-Grams
    #hhhh
    def get_top_ngrams(self, view = 'browser', entry = 'kwp', ngrams = 1, stop_words = [], wordsn = 15):
        sw_full = []
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        if  (entry == 'kwp'):
            corpora = pd.Series([' '.join(k) for k in self.kid]) 
        elif (entry == 'kwa'): 
            corpora = pd.Series([' '.join(a) for a in self.auk] )
        elif (entry == 'abs'):
            corpora = self.data['abstract']
        elif (entry == 'title'):
            corpora = self.data['title']
        if (len(stop_words) > 0):
            for sw_ in stop_words: 
                if   (sw_ == 'ar' or sw_ == 'ara'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Arabic.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Arabic.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'bn' or sw_ == 'ben'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bengali.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Bengali.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'bg' or sw_ == 'bul'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bulgarian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Bulgarian.txt', 'r',     encoding = 'utf8')
                elif (sw_ == 'cs' or sw_ == 'cze' or sw_ == 'ces'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Czech.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Czech.txt', 'r',         encoding = 'utf8')
                elif (sw_ == 'en' or sw_ == 'eng'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-English.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-English.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fi' or sw_ == 'fin'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Finnish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Finnish.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fr' or sw_ == 'fre' or sw_ == 'fra'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-French.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-French.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'de' or sw_ == 'ger' or sw_ == 'deu'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-German.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-German.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'hi' or sw_ == 'hin'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hind.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Hindi.txt', 'r',         encoding = 'utf8')
                elif (sw_ == 'hu' or sw_ == 'hun'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hungarian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Hungarian.txt', 'r',     encoding = 'utf8')
                elif (sw_ == 'it' or sw_ == 'ita'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Italian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Italian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'mr' or sw_ == 'mar'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Marathi.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Marathi.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fa' or sw_ == 'per' or sw_ == 'fas'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Persian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Persian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'pl' or sw_ == 'pol'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Polish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Polish.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'pt-br' or sw_ == 'por-br'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Portuguese-br.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Portuguese-br.txt', 'r', encoding = 'utf8')
                elif (sw_ == 'ro' or sw_ == 'rum' or sw_ == 'ron'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Romanian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Romanian.txt', 'r',      encoding = 'utf8')
                elif (sw_ == 'ru' or sw_ == 'rus'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Russian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Russian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'es' or sw_ == 'spa'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Spanish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Spanish.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'sv' or sw_ == 'swe'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Swedish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Swedish.txt', 'r',       encoding = 'utf8')
                f_lines = f_file.read()
                sw      = f_lines.split('\n')
                sw      = list(filter(None, sw))
                sw_full.extend(sw)
        vec          = CountVectorizer(stop_words = frozenset(sw_full), ngram_range = (ngrams, ngrams)).fit(corpora)
        bag_of_words = vec.transform(corpora)
        sum_words    = bag_of_words.sum(axis = 0)
        words_freq   = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq   = sorted(words_freq, key = lambda x: x[1], reverse = True)
        common_words = words_freq[:wordsn]
        words        = []
        freqs        = []
        for word, freq in common_words:
            words.append(word)
            freqs.append(freq) 
        df  = pd.DataFrame({'Word': words, 'Freq': freqs})
        fig = go.Figure(go.Bar(
                                x           = df['Freq'],
                                y           = df['Word'],
                                orientation = 'h',
                                marker      = dict(color = 'rgba(246, 78, 139, 0.6)', line = dict(color = 'black', width = 1))
                               ),
                        )
        fig.update_yaxes(autorange = 'reversed')
        fig.update_layout(paper_bgcolor = 'rgb(248, 248, 255)', plot_bgcolor = 'rgb(248, 248, 255)')
        fig.show()
        return
    
    # Function: Tree Map
    def tree_map(self, entry = 'kwp', topn = 20, size_x = 10, size_y = 10): 
        if   (entry == 'kwp'):
            labels = self.u_kid
            sizes  = self.kid_count 
            title  = 'Keywords Plus'
        elif (entry == 'kwa'):
            labels = self.u_auk
            sizes  = self.auk_count 
            title  = "Authors' Keywords"
        elif (entry == 'aut'):
            labels = self.u_aut
            sizes  = self.doc_aut
            title  = 'Authors'
        elif (entry == 'jou'):
            labels = self.u_jou
            sizes  = self.jou_count
            title  = 'Sources'
        elif (entry == 'ctr'):
            labels = self.u_ctr
            sizes  = self.ctr_count 
            title  = 'Coutries'
        elif (entry == 'inst'):
            labels = self.u_uni
            sizes  = self.uni_count 
            title  = 'Institutions'
        idx    = sorted(range(len(sizes)), key = sizes.__getitem__)
        idx.reverse()
        labels = [labels[i] for i in idx]
        sizes  = [sizes[i]  for i in idx]
        labels = labels[:topn]
        labels = [labels[i]+'\n ('+str(sizes[i])+')' for i in range(0, len(labels))]
        sizes  = sizes[:topn] 
        cols   = [plt.cm.Spectral(i/float(len(labels))) for i in range(0, len(labels))]
        plt.figure(figsize = (size_x, size_y))
        squarify.plot(sizes = sizes, label = labels, pad = True, color = cols, alpha = 0.75)
        plt.title(title, loc = 'center')
        plt.axis('off')
        plt.show()
        return
    
    # Function: Authors' Productivity Plot
    def authors_productivity(self, view = 'browser', topn = 20): 
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (topn > len(self.u_aut)):
            topn = len(self.u_aut)
        years        = list(range(self.date_str, self.date_end+1))    
        dicty        = dict(zip(years, list(range(0, len(years)))))
        idx          = sorted(range(0, len(self.doc_aut)), key = self.doc_aut.__getitem__)
        idx.reverse()
        key          = [self.u_aut[i] for i in idx]
        key          = key[:topn]
        n_id         = [[ [] for item in years] for item in key]
        productivity = pd.DataFrame(np.zeros((topn, len(years))), index = key, columns = years)
        Xv           = []
        Yv           = []
        Xe           = []
        Ye           = []
        for n in range(0, len(key)):
            name = key[n]
            docs = [i for i in range(0, len(self.aut)) if name in self.aut[i]]
            for i in docs:
                j                       = dicty[int(self.data.loc[i, 'year'])]
                productivity.iloc[n, j] = productivity.iloc[n, j] + 1
                n_id[n][j].append( 'id: '+str(i)+' ('+name+', '+self.data.loc[i, 'year']+')')
                Xv.append(n)
                Yv.append(j)
        node_list_a = [ str(int(productivity.iloc[Xv[i], Yv[i]])) for i in range(0, len(Xv)) ]
        nid_list    = [ n_id[Xv[i]][Yv[i]] for i in range(0, len(Xv)) ]
        nid_list_a  = []
        for item in nid_list:
            if (len(item) == 1):
                nid_list_a.append(item)
            else:
                itens = []
                itens.append(item[0])
                for i in range(1, len(item)):
                    itens[0] = itens[0]+'<br>'+item[i]
                nid_list_a.append(itens)
        nid_list_a = [txt[0] for txt in nid_list_a]
        for i in range(0, len(Xv)-1):
            if (Xv[i] == Xv[i+1]):
                Xe.append(Xv[i]*1.00)
                Xe.append(Xv[i+1]*1.00)
                Xe.append(None)
                Ye.append(Yv[i]*1.00)
                Ye.append(Yv[i+1]*1.00)
                Ye.append(None)
        a_trace = go.Scatter(x         = Ye,
                             y         = Xe,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Yv,
                             y         = Xv,
                             opacity   = 1,
                             mode      = 'markers+text',
                             marker    = dict(symbol = 'circle-dot', size = 25, color = 'purple'),
                             text      = node_list_a,
                             hoverinfo = 'text',
                             hovertext = nid_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#e0e0e0',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True, 
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, len(years))),
                                                  ticktext       = years,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                               ),
                            yaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'grey',
                                                  zeroline       = False, 
                                                  showticklabels = True,
                                                  tickmode       = 'array', 
                                                  tickvals       = list(range(0, topn)),
                                                  ticktext       = key,
                                                  spikedash      = 'solid',
                                                  spikecolor     = 'blue',
                                                  spikethickness = 2
                                                )
                            )
        fig_aut = go.Figure(data = [a_trace, n_trace], layout = layout)
        fig_aut.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_aut.update_yaxes(autorange = 'reversed')
        fig_aut.show() 
        return
    
    # Function: Evolution per Year
    def plot_evolution_year(self, view = 'browser', stop_words = ['en'], key = 'kwp', rmv_custom_words = [], target_word = [], topn = 10, start = 2010, end = 2022):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (start < self.date_str):
            start = self.date_str
        if (end > self.date_end):
            end = self.date_end
        y_idx = [i for i in range(0, self.data.shape[0]) if int(self.data.loc[i, 'year']) >= start and int(self.data.loc[i, 'year']) <= end]
        if (len(rmv_custom_words) == 0):
            rmv_custom_words = ['unknow']
        else:
            rmv_custom_words.append('unknow') 
        if   (key == 'kwp'):
            u_ent, ent = self.u_kid, self.kid
        elif (key == 'kwa'):
            u_ent, ent = self.u_auk, self.auk
        elif (key == 'jou'):
            u_ent, ent = self.u_jou, self.jou
        elif (key == 'abs'):
            abs_  = self.data['abstract'].tolist()
            abs_  = ['the' if i not in y_idx else  abs_[i] for i in range(0, len(abs_))]
            abs_  = self.clear_text(abs_, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
            u_abs = [item.split() for item in abs_]
            u_abs = [item for sublist in u_abs for item in sublist]
            u_abs = list(set(u_abs))
            if (u_abs[0] == ''):
                u_abs = u_abs[1:]
            s_abs       = [item.split() for item in abs_]
            s_abs       = [item for sublist in s_abs for item in sublist]
            abs_count   = [s_abs.count(item) for item in u_abs]
            idx         = sorted(range(len(abs_count)), key = abs_count.__getitem__)
            idx.reverse()
            abs_       = [item.split() for item in abs_]
            u_abs      = [u_abs[i] for i in idx]
            u_ent, ent = u_abs, abs_
        elif (key == 'title'):
            tit_  = self.data['title'].tolist()
            tit_  = ['the' if i not in y_idx else  tit_[i] for i in range(0, len(tit_))]
            tit_  = self.clear_text(tit_, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
            u_tit = [item.split() for item in tit_]
            u_tit = [item for sublist in u_tit for item in sublist]
            u_tit = list(set(u_tit))
            if (u_tit[0] == ''):
                u_tit = u_tit[1:]
            s_tit       = [item.split() for item in tit_]
            s_tit       = [item for sublist in s_tit for item in sublist]
            tit_count   = [s_tit.count(item) for item in u_tit]
            idx         = sorted(range(len(tit_count)), key = tit_count.__getitem__)
            idx.reverse()
            tit_       = [item.split() for item in tit_]
            u_tit      = [u_tit[i] for i in idx]
            u_ent, ent = u_tit, tit_
        traces = []
        years  = list(range(self.date_str, self.date_end+1))
        dict_y = dict(zip(years, list(range(0, len(years)))))
        themes = self.__get_counts_year(u_ent, ent)
        w_idx  = []
        if (len(target_word) > 0):
            posit  = -1
            for word in target_word:
                if (word.lower() in u_ent):
                    posit = u_ent.index(word.lower())
                if (posit > 0):
                    w_idx.append(posit)
            if (len(w_idx) > 0):
                themes = themes.iloc[w_idx, :]
        for j in range(dict_y[start], dict_y[end]+1):
            theme_vec = themes.iloc[:, j]
            theme_vec = theme_vec[theme_vec > 0]
            if (len(theme_vec) > 0):
                theme_vec = theme_vec.sort_values(ascending = False) 
                theme_vec = theme_vec.iloc[:topn] 
                idx       = theme_vec.index.tolist()
                names     = [u_ent[item] for item in idx]
                if (len(w_idx) > 0):
                    values = [themes.loc[item, j] for item in idx]
                else:
                    values = [themes.iloc[item, j] for item in idx]
                n_val     = [names[i]+' ('+str(int(values[i]))+')' for i in range(0, len(names))]
                data      = go.Bar(x                = [years[j]]*len(values), 
                                   y                = values, 
                                   text             = names, 
                                   hoverinfo        = 'text',
                                   textangle        = 0,
                                   textfont_size    = 10,
                                   hovertext        = n_val,
                                   insidetextanchor = 'middle',
                                   marker_color     = self.__hex_rgba(hxc = self.color_names[j], alpha = 0.70)
                                   )
                traces.append(data)
        layout = go.Layout(barmode      = 'stack', 
                           showlegend   = False,
                           hovermode    = 'closest',
                           margin       = dict(b = 10, l = 5, r = 5, t = 10),
                           plot_bgcolor = '#f5f5f5',
                           xaxis        = dict(tickangle      =  35,
                                               showticklabels = True, 
                                               type           = 'category'
                                              )
                           )
        fig = go.Figure(data = traces, layout = layout)
        fig.show()
        return     

    # Function: Plot Bar 
    def plot_bars(self, statistic = 'dpy', topn = 20, size_x = 10, size_y = 10): 
        if  (statistic.lower() == 'dpy'):
            key   = list(range(self.date_str, self.date_end+1))
            value = [self.data[self.data.year == str(item)].shape[0] for item in key]
            title = 'Documents per Year'
            x_lbl = 'Year'
            y_lbl = 'Documents'
        elif(statistic.lower() == 'cpy'):
            key   = list(range(self.date_str, self.date_end+1))
            value = []
            title = 'Citations per Year'
            x_lbl = 'Year'
            y_lbl = 'Citations'
            for i in range(0, len(key)):
                year = key[i]
                idx  = [i for i, x in enumerate(list(self.dy)) if x == year]
                docs = 0
                for j in idx:
                    docs = docs + self.citation[j]
                value.append(docs)
        elif(statistic.lower() == 'ppy'):
            key, value = self.__get_past_citations_year()
            title      = 'Past Citations per Year'
            x_lbl      = 'Year'
            y_lbl      = 'Past Citations'
        elif (statistic.lower() == 'ltk'):
            value = list(range(1, max(self.doc_aut)+1))
            key   = [self.doc_aut.count(item) for item in value]
            idx   = [i for i in range(0, len(key)) if key[i] > 0]
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            title = "Lotka's Law"
            x_lbl = 'Documents'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'spd'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Sources per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'spc'):
            key   = self.u_jou
            value = self.jou_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Sources per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'apd'):
            key   = self.u_aut
            value = self.doc_aut
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'apc'):
            key   = self.u_aut
            value = self.aut_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'aph'):
            key   = self.u_aut
            value = self.aut_h
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Authors per H-Index'
            x_lbl = 'H-Index'
            y_lbl = 'Authors'
        elif (statistic.lower() == 'bdf_1'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] <= c1]
            value = [value[i] for i in range(0, len(value)) if value[i] <= c1]
            title = "Bradford's Law - Core Sources 1"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'bdf_2'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            c2    = int(value[-1]*(2/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] > c1 and value[i] <= c2]
            value = [value[i] for i in range(0, len(value)) if value[i] > c1 and value[i] <= c2]
            title = "Bradford's Law - Core Sources 2"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'bdf_3'):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c2    = int(value[-1]*(2/3))
            key   = [key[i]   for i in range(0, len(key))   if value[i] > c2]
            value = [value[i] for i in range(0, len(value)) if value[i] > c2]
            title = "Bradford's Law - Core Sources 3"
            x_lbl = 'Documents'
            y_lbl = 'Sources'
        elif (statistic.lower() == 'ipd'):
            key   = self.u_uni
            value = self.uni_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Institutions per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Institutions'
        elif (statistic.lower() == 'ipc'):
            key   = self.u_uni
            value = self.uni_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Institutions per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Institutions'
        elif (statistic.lower() == 'cpd'):
            key   = self.u_ctr
            value = self.ctr_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Countries per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Countries'
        elif (statistic.lower() == 'cpc'):
            key   = self.u_ctr
            value = self.ctr_cit
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Countries per Citations'
            x_lbl = 'Citations'
            y_lbl = 'Countries'
        elif (statistic.lower() == 'lpd'):
            key   = self.u_lan
            value = self.lan_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Language per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Languages'
        elif (statistic.lower() == 'kpd'):
            key   = self.u_kid
            value = self.kid_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+' - Keywords Plus per Documents'
            x_lbl = 'Documents'
            y_lbl = 'Keywords Plus'
        elif (statistic.lower() == 'kad'):
            key   = self.u_kid
            value = self.kid_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            key   = key[:topn]
            value = value[:topn]
            title = 'Top '+ str(topn)+" - Authors' Keywords per Documents"
            x_lbl = 'Documents'
            y_lbl = "Authors' Keywords"
        w_1 = 0.135
        w_2 = 0.045
        w_s = np.arange(len(value) / 8, step = 0.125)
        plt.figure(figsize = [size_x, size_y])
        if (statistic.lower() == 'dpy' or statistic.lower() == 'cpy' or statistic.lower() == 'ppy' or statistic.lower() == 'ltk'):
            plt.bar(w_s, value, color = '#dce6f2', width = w_1/2, edgecolor = '#c3d5e8')
            plt.bar(w_s, value, color = '#ffc001', width = w_2/2, edgecolor = '#c3d5e8')
            if (statistic.lower() != 'ltk'):
                plt.axhline(y = sum(value)/len(value), color = 'r', linestyle = '-', lw = 0.75)
            plt.axhline(y = 0, color = 'gray')
            plt.xticks(w_s, key)
            plt.xticks(rotation = 90)
            for i, bar in enumerate(value):
                plt.text(x = i / 8 - 0.015, y = bar + 0.5, s = bar)
        else:
            plt.barh(key, color = '#dce6f2', width = value, height = w_1*5, edgecolor = '#c3d5e8')
            plt.barh(key, color = '#ffc001', width = value, height = w_2*5, edgecolor = '#c3d5e8')
            plt.yticks(key)
            for i, bar in enumerate(value):
                plt.text(x = bar + 0.05, y = key[i], s = bar)
            plt.gca().invert_yaxis()
        plt.title(title, loc = 'center')
        plt.xlabel(x_lbl)
        plt.ylabel(y_lbl)
        plt.show()
        return
    
    # Function: Sankey Diagram
    def sankey_diagram(self, view = 'browser', entry = ['aut', 'cout', 'inst'], topn = 20): 
        def sort_count(u_lst, count_lst):
            idx   = sorted(range(0, len(count_lst)), key = count_lst.__getitem__)
            idx.reverse()
            u_lst = [u_lst[i] for i in idx]
            return u_lst   
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        u_keys = ['aut', 'cout', 'inst', 'jou', 'kwa', 'kwp', 'lan']
        u_name = ['Authors', 'Countries', 'Institutions', 'Journals', 'Auhors_Keywords', 'Keywords_Plus', 'Languages']
        u_idx  = [i for i in range(0, len(u_keys)) if u_keys[i] == entry[0]]
        u_cnt  = [self.doc_aut, self.ctr_count, self.uni_count, self.jou_count, self.auk_count, self.kid_count, self.lan_count]
        u_list = [self.aut, self.ctr, self.uni, self.jou, self.auk, self.kid, self.lan]
        u_sort = [self.u_aut, self.u_ctr, self.u_uni, self.u_jou, self.u_auk, self.u_kid, self.u_lan]
        u_sort = [u_sort[i] if i != u_idx[0] else sort_count(u_sort[i], u_cnt[i]) for i in range(0, len(u_sort)) ]
        dict_e = dict( zip( u_keys, u_list ) )
        dict_u = dict( zip( u_keys, u_sort ) )
        dict_n = dict( zip( u_keys, u_name ) )
        list_e = []
        list_u = []
        sk_s   = []
        sk_t   = []
        sk_v   = []
        for e in entry:
            list_e.append(dict_e[e])
            list_u.append(dict_u[e])
            if ( e == entry[0] ):
                start  = [0]
                finish = [len(list_u[-1])-1]
            else:
                start.append(finish[-1]+1)
                finish.append(start[-1]+len(list_u[-1])-1)
        label  = [item for sublist in list_u for item in sublist if item != 'UNKNOW']
        for i in range(0, len(entry)):
            label.append('UNKNOW_'+dict_n[entry[i]])
        dict_s = dict(  zip( label, list( range( 0, len(label) ) ) ) )
        pairs  = [[] for _ in range(0, len(list_u)-1)]
        for i in range(0, len(list_u)-1):
            j = i + 1
            for m in range(0, len(list_u[i])):
                name_a = list_u[i][m]
                idx_a  = [k for k in range(0, len(list_e[i])) if name_a in list_e[i][k] ]
                name_a = name_a.replace('UNKNOW', 'UNKNOW_'+dict_n[entry[j-1]])
                for n in idx_a:
                    if (entry[i] != 'kwa' and entry[i] != 'kwp'):
                        pos_a  = list_e[i][n].index(name_a.replace('UNKNOW_'+dict_n[entry[j-1]], 'UNKNOW'))
                        pos_a  = np.clip(pos_a, 0, len(list_e[j][n])-1)
                    else:
                        pos_a  = 0
                    if (entry[j] != 'kwa' and entry[j] != 'kwp'):
                        if (len(list_e[j][n]) > 0):
                            name_b = list_e[j][n][pos_a].replace('UNKNOW', 'UNKNOW_'+dict_n[entry[j]])
                        else:
                            name_b = 'UNKNOW_'+dict_n[entry[j]]
                        pairs[i].append([name_a, name_b])
                    else:
                        name_b = list_e[j][n]
                        for name in name_b:
                            pairs[i].append([name_a, name.replace('UNKNOW', 'UNKNOW_'+dict_n[entry[j]])])
        for pair in pairs:
            if (pair == pairs[0]):
                u_pair = list(set([tuple(x) for x in pair]))
                u_pair = [ [ item[0], item[1] ] for item in u_pair ]
                u_vals = Counter(str(elem) for elem in pair)
                u_vals = [u_vals[str(i)] for i in u_pair]
                idx    = list(np.argsort(-np.array(u_vals), kind = 'stable'))
                s_pair = [u_pair[i] for i in idx]
                s_vals = [u_vals[i] for i in idx]
                topn   = np.clip(topn, 0, len(s_vals))
                u_next = [item[1] for item in s_pair]
                u_next = u_next[:topn]
                for i in range(0, topn):
                    if (s_pair[i][0] != '' and s_pair[i][1] != ''):
                        sk_s.append(dict_s[s_pair[i][0]])
                        sk_t.append(dict_s[s_pair[i][1]])
                        sk_v.append(s_vals[i])
            else:
                u_pair = list(set([tuple(x) for x in pair]))
                u_pair = [[ item[0], item[1] ] for item in u_pair]
                select = []
                for p in range(0, len(u_pair)):
                    b, _ = u_pair[p]
                    if (b in u_next):
                        select.append(b)
                select = list(set(select))
                for p in range(len(u_pair)-1, -1, -1):
                    b, _ = u_pair[p]
                    if (b not in select):
                        del u_pair[p]
                u_vals = Counter(str(elem) for elem in pair)
                u_vals = [u_vals[str(i)] for i in u_pair]
                idx    = list(np.argsort(-np.array(u_vals), kind = 'stable'))
                s_pair = [u_pair[i] for i in idx]
                s_vals = [u_vals[i] for i in idx]
                topn   = np.clip(topn, 0, len(s_vals))
                u_next = [ item[1] for item in s_pair]
                u_next = u_next[:topn]
                for i in range(0, topn):
                    if (s_pair[i][0] != '' and s_pair[i][1] != ''):
                        sk_s.append(dict_s[s_pair[i][0]])
                        sk_t.append(dict_s[s_pair[i][1]])
                        sk_v.append(s_vals[i])             
        if (len(sk_s) > len(self.color_names)):
            count = 0
            while (len(self.color_names) < len(sk_s)):
                self.color_names.append(self.color_names[count])
                count = count + 1
        link = dict(source = sk_s, target = sk_t,   value = sk_v, color = self.color_names)
        node = dict(label  = label,   pad = 10, thickness = 15,   color = 'white')
        data = go.Sankey(
                          link        = link, 
                          node        = node, 
                          arrangement = 'freeform'
                         )
        fig  = go.Figure(data)
        nt   = 'Sankey Diagram ( '
        for e in range(0, len(entry)):
            nt = nt + str(dict_n[entry[e]]) + ' / '
        nt = nt[:-2] + ')'
        fig.update_layout(hovermode = 'closest', title = nt, font = dict(size = 12, color = 'white'), paper_bgcolor = '#474747')
        fig.show()
        return

    #############################################################################
    
    # Function: Hirsch Index
    def __h_index(self):
        h_i = []
        for researcher in self.u_aut:
            doc = []
            i   = 0
            for researchers in self.aut:
                if (researcher in researchers):
                    doc.append(self.citation[i])
                i = i + 1
            for j in range(len(doc)-1, -1, -1):
                count = len([element for element in doc if element >= j])
                if (count >= j):
                    h_i.append(j)
                    break
        return h_i
    
    # Function: Total and Self Citations
    def __total_and_self_citations(self):
        t_c = []
        s_c = []
        for researcher in self.u_aut:
            doc = []
            cit = 0
            i1  = 0
            i2  = 0
            for researchers in self.aut:
                if (researcher in researchers):
                    doc.append(self.citation[i1])
                    for reference in self.ref[i2]:
                        if (researcher in reference.lower()):
                            cit = cit + 1
                i1 = i1 + 1
            i2 = i2 + 1
            t_c.append(sum(doc))
            s_c.append(cit)
        return t_c, s_c

    #############################################################################

    # Function: Text Pre-Processing
    def clear_text(self, corpus, stop_words = ['en'], lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = [], verbose = False):
        sw_full = []
        # Lower Case
        if (lowercase == True):
            if (verbose == True):
                print('Lower Case: Working...')
            corpus = [str(x).lower() for x in  corpus]
            if (verbose == True):
                print('Lower Case: Done!')
        # Replace Accents 
        if (rmv_accents == True):
            if (verbose == True):
                print('Removing Accents: Working...')
            for i in range(0, len(corpus)):
                text = corpus[i]
                try:
                    text = unicode(text, 'utf-8')
                except NameError: 
                    pass
                text      = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
                corpus[i] = str(text)
            if (verbose == True):
                print('Removing Accents: Done!')
        # Remove Punctuation & Special Characters
        if (rmv_special_chars == True):
            if (verbose == True):
                print('Removing Special Characters: Working...')
            corpus = [re.sub(r"[^a-zA-Z0-9]+", ' ', i) for i in corpus]
            if (verbose == True):
                print('Removing Special Characters: Done!')
        # Remove Numbers
        if (rmv_numbers == True):
            if (verbose == True):
                print('Removing Numbers: Working...')
            corpus = [re.sub('[0-9]', ' ', i) for i in corpus] 
            if (verbose == True):
                print('Removing Numbers: Done!')
        # Remove Stopwords
        if (len(stop_words) > 0):
            for sw_ in stop_words: 
                if   (sw_ == 'ar' or sw_ == 'ara'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Arabic.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Arabic.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'bn' or sw_ == 'ben'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bengali.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Bengali.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'bg' or sw_ == 'bul'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Bulgarian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Bulgarian.txt', 'r',     encoding = 'utf8')
                elif (sw_ == 'cs' or sw_ == 'cze' or sw_ == 'ces'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Czech.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Czech.txt', 'r',         encoding = 'utf8')
                elif (sw_ == 'en' or sw_ == 'eng'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-English.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-English.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fi' or sw_ == 'fin'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Finnish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Finnish.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fr' or sw_ == 'fre' or sw_ == 'fra'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-French.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-French.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'de' or sw_ == 'ger' or sw_ == 'deu'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-German.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-German.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'hi' or sw_ == 'hin'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hind.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Hindi.txt', 'r',         encoding = 'utf8')
                elif (sw_ == 'hu' or sw_ == 'hun'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Hungarian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Hungarian.txt', 'r',     encoding = 'utf8')
                elif (sw_ == 'it' or sw_ == 'ita'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Italian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Italian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'mr' or sw_ == 'mar'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Marathi.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Marathi.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'fa' or sw_ == 'per' or sw_ == 'fas'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Persian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Persian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'pl' or sw_ == 'pol'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Polish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Polish.txt', 'r',        encoding = 'utf8')
                elif (sw_ == 'pt-br' or sw_ == 'por-br'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Portuguese-br.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Portuguese-br.txt', 'r', encoding = 'utf8')
                elif (sw_ == 'ro' or sw_ == 'rum' or sw_ == 'ron'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Romanian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Romanian.txt', 'r',      encoding = 'utf8')
                elif (sw_ == 'ru' or sw_ == 'rus'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Russian.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Russian.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'es' or sw_ == 'spa'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Spanish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Spanish.txt', 'r',       encoding = 'utf8')
                elif (sw_ == 'sv' or sw_ == 'swe'):
                    f_file = pkg_resources.open_text(stws, 'Stopwords-Swedish.txt', encoding = 'utf8')
                    #f_file = open('../pyBibX/Stopwords-Swedish.txt', 'r',       encoding = 'utf8')
                f_lines = f_file.read()
                sw      = f_lines.split('\n')
                sw      = list(filter(None, sw))
                sw_full.extend(sw)
            if (verbose == True):
                print('Removing Stopwords: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x for x in text if x not in sw_full]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Stopwords: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Stopwords: Done!')
        # Remove Custom Words
        if (len(rmv_custom_words) > 0):
            if (verbose == True):
                print('Removing Custom Words: Working...')
            for i in range(0, len(corpus)):
               text      = corpus[i].split()
               text      = [x for x in text if x not in rmv_custom_words]
               corpus[i] = ' '.join(text) 
               if (verbose == True):
                   print('Removing Custom Words: ' + str(i + 1) +  ' of ' + str(len(corpus)) )
            if (verbose == True):
                print('Removing Custom Word: Done!')
        for i in range(0, len(corpus)):
            corpus[i] = ' '.join(corpus[i].split())
        return corpus

    # Function: TF-IDF
    def dtm_tf_idf(self, corpus):
        vectorizer = TfidfVectorizer(norm = 'l2')
        tf_idf     = vectorizer.fit_transform(corpus)
        try:
            tokens = vectorizer.get_feature_names_out()
        except:
            tokens = vectorizer.get_feature_names()
        values     = tf_idf.todense()
        values     = values.tolist()
        dtm        = pd.DataFrame(values, columns = tokens)
        return dtm
   
    # Function: Projection
    def docs_projection(self, view = 'browser', corpus_type = 'abs', stop_words = ['en'], rmv_custom_words = [], custom_label = [], custom_projection = [], n_components = 2, n_clusters = 5, tf_idf = True, embeddings = False, method = 'tsvd'):
        if   (corpus_type == 'abs'):
            corpus = self.data['abstract']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'title'):
            corpus = self.data['title']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'kwa'): 
            corpus = self.data['author_keywords']
            corpus = corpus.tolist()
        elif (corpus_type == 'kwp'):
            corpus = self.data['keywords']
            corpus = corpus.tolist()
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (embeddings == True):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embds = model.encode(corpus)
        if (tf_idf == True):
            dtm = self.dtm_tf_idf(corpus)
        if (method.lower() == 'umap'):
            decomposition = UMAP(n_components = n_components, random_state = 1001)
        else:
            decomposition = tsvd(n_components = n_components, random_state = 1001)
        if (len(custom_projection) == 0 and embeddings == False):
            transformed = decomposition.fit_transform(dtm)
        elif (len(custom_projection) == 0 and embeddings == True):
            transformed = decomposition.fit_transform(embds)
        elif (custom_projection.shape[0] == self.data.shape[0] and custom_projection.shape[1] >= 2):
            transformed = np.copy(custom_projection)
        if (len(custom_label) == 0):
            cluster = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 100, max_iter = 10, random_state = 1001)
            if (tf_idf == True and embeddings == False):
                cluster.fit(dtm)
            else:
                cluster.fit(transformed)
            labels  = cluster.labels_
            n       = len(set(labels.tolist()))
        elif (len(custom_label) > 0):
            labels = [item for item in custom_label]
            n      = len(set(labels))
        n_trace = []
        for i in range(0, n):
            labels_c    = []
            node_list   = []
            #node_list_h = []
            n_id        = []
            #n_id_h      = []
            #x_h         = []
            #y_h         = []
            idx         = [j for j in range(0, len(labels)) if labels[j] == i]
            #idx_h       = []
            x           = transformed[idx, 0]
            y           = transformed[idx, 1]
            #if (hull_plot == True):
                #pts = np.c_[x,y]
                #try:
                    #hull  = ConvexHull(pts)
                    #pts_h = pts[hull.vertices, :]
                    #pts_  = [ ( pts[i, 0], pts[i, 1] ) for i in range(0, pts.shape[0])]
                    #x     = [item[0] for item in pts_]
                    #y     = [item[1] for item in pts_]
                    #x_h.extend(pts_h[:,0].tolist())
                    #y_h.extend(pts_h[:,1].tolist())
                    #for hv in hull.vertices:
                        #node_list_h.etend(idx[hv])
                        #idx_h.extend(idx[hv])
                #except:
                    #hull_plot = False
            labels_c.extend(self.color_names[i] for item in idx)
            node_list.extend(idx)
            for j in range(0, len(idx)):
                n_id.append(
                            'id:' +str(idx[j])               +'<br>'  +
                            'cluster:' +str(i)               +'<br>'  +
                             self.data.loc[idx[j], 'author'] +' ('    +
                             self.data.loc[idx[j], 'year']   +'). '   +
                             self.data.loc[idx[j], 'title']  +'. '    +
                             self.data.loc[idx[j], 'journal']+'. doi:'+
                             self.data.loc[idx[j], 'doi']    +'.'
                             )
                n_id[-1] = '<br>'.join(textwrap.wrap(n_id[-1], width = 50))
            #if (hull_plot == True):
                #for j in range(0, len(idx_h)):
                    #n_id_h.append(
                                #'id:' +str(idx_h[j])               +'<br>'  +
                                #'cluster:' +str(i)                 +'<br>'  +
                                # self.data.loc[idx_h[j], 'author'] +' ('    +
                                # self.data.loc[idx_h[j], 'year']   +'). '   +
                                # self.data.loc[idx_h[j], 'title']  +'. '    +
                                # self.data.loc[idx_h[j], 'journal']+'. doi:'+
                                # self.data.loc[idx_h[j], 'doi']    +'.'
                                # )
                    #n_id_h[-1] = '<br>'.join(textwrap.wrap(n_id_h[-1], width = 50))
            #if (hull_plot == True):
                #n_trace.append(go.Scatter(x         = x_h,
                                          #y         = y_h,
                                          #opacity   = 1,
                                          #mode      = 'markers+text',
                                          #marker    = dict(symbol = 'circle-dot', size = 25, color = self.color_names[i]),
                                          #fill      = 'toself',
                                          #fillcolor = self.__hex_rgba(hxc = self.color_names[i], alpha = 0.15),
                                          #text      = node_list_h,
                                          #hoverinfo = 'text',
                                          #hovertext = n_id_h,
                                          #name      = ''
                                          #))
            n_trace.append(go.Scatter(x         = x,
                                      y         = y,
                                      opacity   = 1,
                                      mode      = 'markers+text',
                                      marker    = dict(symbol = 'circle-dot', size = 25, color = self.color_names[i]),
                                      text      = node_list,
                                      hoverinfo = 'text',
                                      hovertext = n_id,
                                      name      = ''
                                      ))
            
        layout  = go.Layout(showlegend   = False,
                            hovermode    = 'closest',
                            margin       = dict(b = 10, l = 5, r = 5, t = 10),
                            plot_bgcolor = '#f5f5f5',
                            xaxis        = dict(  showgrid       = True, 
                                                  gridcolor      = 'white',
                                                  zeroline       = False, 
                                                  showticklabels = False, 
                                               ),
                            yaxis        = dict(  showgrid       = True,  
                                                  gridcolor      = 'white',
                                                  zeroline       = False, 
                                                  showticklabels = False,
                                                )
                            )
        fig_proj = go.Figure(data = n_trace, layout = layout)
        fig_proj.update_traces(textfont_size = 10, textfont_color = 'white') 
        fig_proj.show() 
        return transformed, labels

    #############################################################################
    
    # Function: Authors Colaboration Adjacency Matrix
    def __adjacency_matrix_aut(self, min_colab = 1):
        self.matrix_a = pd.DataFrame(np.zeros( (len(self.u_aut), len(self.u_aut))), index = self.u_aut, columns = self.u_aut)
        for i in range(0, len(self.aut)):
            if (len(self.aut[i]) > 1):
                for j in range(0, len(self.aut[i]) -1):
                    j1 = j
                    j2 = j+1
                    self.matrix_a.loc[self.aut[i][j1], self.aut[i][j2]] = self.matrix_a.loc[self.aut[i][j1], self.aut[i][j2]] + 1
                    self.matrix_a.loc[self.aut[i][j2], self.aut[i][j1]] = self.matrix_a.loc[self.aut[i][j2], self.aut[i][j1]] + 1
        self.labels_a = ['a_'+str(i) for i in range(0, self.matrix_a.shape[0])]
        self.matrix_a[self.matrix_a > 0] = 1
        self.n_colab  = self.matrix_a.sum(axis = 0)
        self.n_colab  = [int(item) for item in self.n_colab]
        if (min_colab > 0):
            cols =  [i for i in range(0, len(self.n_colab)) if self.n_colab[i] < min_colab]
            if (len(cols) > 0):
                self.matrix_a.iloc[cols, cols] = 0
        return self
    
    # Function: Country Colaboration Adjacency Matrix
    def __adjacency_matrix_ctr(self, min_colab = 1):
        self.matrix_a = pd.DataFrame(np.zeros( (len(self.u_ctr), len(self.u_ctr))), index = self.u_ctr, columns = self.u_ctr)
        for i in range(0, len(self.ctr)):
            if (len(self.ctr[i]) > 1):
                for j in range(0, len(self.ctr[i]) -1):
                    j1 = j
                    j2 = j+1
                    self.matrix_a.loc[self.ctr[i][j1], self.ctr[i][j2]] = self.matrix_a.loc[self.ctr[i][j1], self.ctr[i][j2]] + 1
                    self.matrix_a.loc[self.ctr[i][j2], self.ctr[i][j1]] = self.matrix_a.loc[self.ctr[i][j2], self.ctr[i][j1]] + 1
        np.fill_diagonal(self.matrix_a.values, 0)
        self.labels_a = ['c_'+str(i) for i in range(0, self.matrix_a.shape[0])]
        self.matrix_a[self.matrix_a > 0] = 1
        self.n_colab  = self.matrix_a.sum(axis = 0)
        self.n_colab  = [int(item) for item in self.n_colab]
        if (min_colab > 0):
            cols =  [i for i in range(0, len(self.n_colab)) if self.n_colab[i] < min_colab]
            if (len(cols) > 0):
                self.matrix_a.iloc[cols, cols] = 0
        return self
    
    # Function: Institution Colaboration Adjacency Matrix
    def __adjacency_matrix_inst(self, min_colab = 1):
        self.matrix_a = pd.DataFrame(np.zeros( (len(self.u_uni), len(self.u_uni))), index = self.u_uni, columns = self.u_uni)
        for i in range(0, len(self.uni)):
            if (len(self.uni[i]) > 1):
                for j in range(0, len(self.uni[i]) -1):
                    j1 = j
                    j2 = j+1
                    self.matrix_a.loc[self.uni[i][j1], self.uni[i][j2]] = self.matrix_a.loc[self.uni[i][j1], self.uni[i][j2]] + 1
                    self.matrix_a.loc[self.uni[i][j2], self.uni[i][j1]] = self.matrix_a.loc[self.uni[i][j2], self.uni[i][j1]] + 1
        np.fill_diagonal(self.matrix_a.values, 0)
        self.labels_a = ['i_'+str(i) for i in range(0, self.matrix_a.shape[0])]
        self.matrix_a[self.matrix_a > 0] = 1
        self.n_colab  = self.matrix_a.sum(axis = 0)
        self.n_colab  = [int(item) for item in self.n_colab]
        if (min_colab > 0):
            cols =  [i for i in range(0, len(self.n_colab)) if self.n_colab[i] < min_colab]
            if (len(cols) > 0):
                self.matrix_a.iloc[cols, cols] = 0
        return self
    
    # Function: KWA Colaboration Adjacency Matrix
    def __adjacency_matrix_kwa(self, min_colab = 1):
        self.matrix_a = pd.DataFrame(np.zeros( (len(self.u_auk), len(self.u_auk))), index = self.u_auk, columns = self.u_auk)
        for i in range(0, len(self.auk)):
            if (len(self.auk[i]) > 1):
                for j in range(0, len(self.auk[i])-1):
                    j1 = j
                    j2 = j+1
                    self.matrix_a.loc[self.auk[i][j1], self.auk[i][j2]] = self.matrix_a.loc[self.auk[i][j1], self.auk[i][j2]] + 1
                    self.matrix_a.loc[self.auk[i][j2], self.auk[i][j1]] = self.matrix_a.loc[self.auk[i][j2], self.auk[i][j1]] + 1
        np.fill_diagonal(self.matrix_a.values, 0)
        self.labels_a =  [self.dict_kwa_id[item] for item in list(self.matrix_a.columns)] 
        self.matrix_a[self.matrix_a > 0] = 1
        self.n_colab  = self.matrix_a.sum(axis = 0)
        self.n_colab  = [int(item) for item in self.n_colab]
        if (min_colab > 0):
            cols =  [i for i in range(0, len(self.n_colab)) if self.n_colab[i] < min_colab]
            if (len(cols) > 0):
                self.matrix_a.iloc[cols, cols] = 0
        return self
    
    # Function: KWP Colaboration Adjacency Matrix
    def __adjacency_matrix_kwp(self, min_colab = 1):
        self.matrix_a = pd.DataFrame(np.zeros( (len(self.u_kid), len(self.u_kid))), index = self.u_kid, columns = self.u_kid)
        for i in range(0, len(self.kid)):
            if (len(self.kid[i]) > 1):
                for j in range(0, len(self.kid[i])-1):
                    j1 = j
                    j2 = j+1
                    self.matrix_a.loc[self.kid[i][j1], self.kid[i][j2]] = self.matrix_a.loc[self.kid[i][j1], self.kid[i][j2]] + 1
                    self.matrix_a.loc[self.kid[i][j2], self.kid[i][j1]] = self.matrix_a.loc[self.kid[i][j2], self.kid[i][j1]] + 1
        np.fill_diagonal(self.matrix_a.values, 0)
        self.labels_a =  [self.dict_kwp_id[item] for item in list(self.matrix_a.columns)]
        self.matrix_a[self.matrix_a > 0] = 1
        self.n_colab  = self.matrix_a.sum(axis = 0)
        self.n_colab  = [int(item) for item in self.n_colab]
        if (min_colab > 0):
            cols =  [i for i in range(0, len(self.n_colab)) if self.n_colab[i] < min_colab]
            if (len(cols) > 0):
                self.matrix_a.iloc[cols, cols] = 0
        return self
    
    # Function: References Adjacency Matrix
    def __adjacency_matrix_ref(self, min_cites = 2, local_nodes = False):
        self.matrix_r = pd.DataFrame(np.zeros( (self.data.shape[0], len(self.u_ref))), columns = self.u_ref)
        for i in range(0, self.data.shape[0]):
            for j in range(0, len(self.ref[i])):
                try:
                    k = self.u_ref.index(self.ref[i][j])
                    self.matrix_r.iloc[i, k] = self.matrix_r.iloc[i, k] + 1
                except:
                    pass      
        self.labels_r = ['r_'+str(i) for i in range(0, self.matrix_r.shape[1])]
        if   (self.data_base == 'scopus'):
            keys = self.data['title'].tolist()
            keys = [item.lower().replace('[','').replace(']','') for item in keys]
        elif (self.data_base == 'wos'):
            keys = self.data['doi'].tolist()
            keys = [item.lower() for item in keys]
        insd_r = []
        insd_t = []
        corp   = []
        if (len(self.u_ref) > 0):
            corp.append(self.u_ref[0].lower())
            for i in range(1, len(self.u_ref)):
                corp[-1] = corp[-1]+' '+self.u_ref[i].lower()
            idx_   = [i for i in range(0, len(keys)) if re.search(keys[i], corp[0]) ]
            for i in idx_:
                for j in range(0, len(self.u_ref)):
                    if (re.search(keys[i], self.u_ref[j].lower()) ):
                        insd_r.append('r_'+str(j))
                        insd_t.append(str(i))
                        self.dy_ref[j] = int(self.dy[i])
                        break
        self.dict_lbs = dict(zip(insd_r, insd_t))
        for item in self.labels_r:
            if item not in self.dict_lbs.keys():
                self.dict_lbs[item] = item
        self.labels_r         = [self.dict_lbs[item] for item in self.labels_r]
        self.matrix_r.columns = self.labels_r 
        if (local_nodes == True):
            cols                  = self.matrix_r.columns.values.tolist()
            cols                  = [i for i in range(0, len(cols)) if cols[i].find('r_') == -1]
            self.matrix_r         = self.matrix_r.iloc[:, cols]
            self.labels_r         = [self.labels_r[item] for item in cols]
            self.matrix_r.columns = self.labels_r 
        if (min_cites >= 1):
            cols                  = self.matrix_r.sum(axis = 0).tolist()
            cols                  = [i for i in range(0, len(cols)) if cols[i] >= min_cites]
            self.matrix_r         = self.matrix_r.iloc[:, cols]
            self.labels_r         = [self.labels_r[item] for item in cols]
            self.matrix_r.columns = self.labels_r 
        return self

    # Function: Network Similarities 
    def network_sim(self, view = 'browser', sim_type = 'coup', node_size = -1, node_labels = False, cut_coup = 0.3, cut_cocit = 5):
        if   (sim_type == 'coup'):
            cut = cut_coup
        elif (sim_type == 'cocit'):
            cut = cut_cocit
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True):
            mode = 'markers+text'
            size = 17
        else:
            mode = 'markers'
            size = 10
        if (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        self.__adjacency_matrix_ref(1, False)
        adjacency_matrix = self.matrix_r.values
        if (sim_type == 'coup'):
            adjacency_matrix = cosine_similarity(adjacency_matrix)
        elif (sim_type == 'cocit'):
            x = pd.DataFrame(np.zeros((adjacency_matrix.shape[0], adjacency_matrix.shape[0])))
            for i in range(0, adjacency_matrix.shape[0]):
                for j in range(0, adjacency_matrix.shape[0]):
                    if (i != j):
                        ct = adjacency_matrix[i,:] + adjacency_matrix[j,:]
                        ct = ct.tolist()
                        vl = ct.count(2.0)
                        if (vl > 0):
                            x.iloc[i, j] = vl
            adjacency_matrix = x
        adjacency_matrix = np.triu(adjacency_matrix, k = 1)
        S                = nx.Graph()
        rows, cols       = np.where(adjacency_matrix >= cut)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        u_rows           = list(set(rows.tolist()))
        u_rows           = [str(item) for item in u_rows]
        u_rows           = sorted(u_rows, key = self.natsort)
        u_cols           = list(set(cols.tolist()))
        u_cols           = [str(item) for item in u_cols]
        u_cols           = sorted(u_cols, key = self.natsort)
        for name in u_rows:
            color = 'blue'
            year  = int(self.dy[ int(name) ])
            n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
            S.add_node(name, color = color, year = year, n_id = n_id )
        for name in u_cols:
            if (name not in u_rows):
                color = 'blue'
                year  = int(self.dy[ int(name) ])
                n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
                S.add_node(name, color = color, year = year, n_id = n_id )
        self.sim_table = pd.DataFrame(np.zeros((len(edges), 2)), columns = ['Pair Node', 'Sim('+sim_type+')'])
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = str(srt)
            end_     = str(end)
            if ( end_ != '-1' ):
                wght = round(adjacency_matrix[srt, end], 3)
                S.add_edge(srt_, end_, weight = wght)
                self.sim_table.iloc[i, 0] = '('+srt_+','+end_+')'
                self.sim_table.iloc[i, 1] = wght
        generator      = nx.algorithms.community.girvan_newman(S)
        community      = next(generator)
        community_list = sorted(map(sorted, community))
        for com in community_list:
            community_list.index(com)
            for node in com:
                S.nodes[node]['color'] = self.color_names[community_list.index(com)]
                S.nodes[node]['n_cls'] = community_list.index(com)
        color       = [S.nodes[n]['color'] for n in S.nodes()]
        pos_s       = nx.spring_layout(S, seed = 42, scale = 1)
        node_list_s = list(S.nodes)
        edge_list_s = list(S.edges)
        nids_list_s = [S.nodes[n]['n_id'] for n in S.nodes()]
        nids_list_s = ['<br>'.join(textwrap.wrap(txt, width = 50)) for txt in nids_list_s]
        nids_list_s = ['id: '+node_list_s[i]+'<br>'+nids_list_s[i] for i in range(0, len(nids_list_s))]
        Xw          = [pos_s[k][0] for k in node_list_s]
        Yw          = [pos_s[k][1] for k in node_list_s]
        Xi          = []
        Yi          = []
        for edge in edge_list_s:
            Xi.append(pos_s[edge[0]][0]*1.00)
            Xi.append(pos_s[edge[1]][0]*1.00)
            Xi.append(None)
            Yi.append(pos_s[edge[0]][1]*1.00)
            Yi.append(pos_s[edge[1]][1]*1.00)
            Yi.append(None)
        a_trace = go.Scatter(x         = Xi,
                             y         = Yi,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = Xw,
                             y         = Yw,
                             opacity   = 0.45,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = node_list_s,
                             hoverinfo = 'text',
                             hovertext = nids_list_s,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        fig_s = go.Figure(data = [n_trace, a_trace], layout = layout)
        fig_s.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        fig_s.update_traces(textfont_size = 10, textfont_color = 'blue', textposition = 'top center') 
        fig_s.show()  
        return

    # Function: Map from Country Adjacency Matrix
    def network_adj_map(self, view = 'browser', connections = True, country_lst = []):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        lat_  = [self.country_lat_long[i][0] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        lon_  = [self.country_lat_long[i][1] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        iso_3 = [self.country_alpha_3[i] for i in range(0, len(self.country_lat_long)) if self.country_names[i] in self.u_ctr]
        text  = [item for item in self.country_names if item in self.u_ctr]
        self.__adjacency_matrix_ctr(1)
        adjacency_matrix = self.matrix_a.values
        vals  = [ int(self.dict_ctr_id[text[i]].replace('c_','')) for i in range(0, len(text)) ]
        vals  = [ int(np.sum(adjacency_matrix[i,:])) for i in vals ]
        lat_  = [ lat_[i] for i in range(0, len(vals)) if vals[i] > 0]
        lon_  = [ lon_[i] for i in range(0, len(vals)) if vals[i] > 0]
        iso_3 = [iso_3[i] for i in range(0, len(vals)) if vals[i] > 0]
        text  = [ text[i] for i in range(0, len(vals)) if vals[i] > 0]
        vals  = [ vals[i] for i in range(0, len(vals)) if vals[i] > 0]
        rows, cols = np.where(adjacency_matrix >= 1)
        edges      = list(zip(rows.tolist(), cols.tolist()))
        nids_list  = ['id:                        ' +self.dict_ctr_id[text[i]]+'<br>'+
                      'country:               '     +text[i].upper()+'<br>' +
                      'collaborators:      '        +str(vals[i])  
                      for i in range(0, len(lat_))]
        Xa         = []
        Ya         = []
        Xb         = []
        Yb         = []
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt      = 'c_'+str(srt)
            end      = 'c_'+str(end)
            srt      = self.dict_id_ctr[srt]
            end      = self.dict_id_ctr[end]
            if (len(country_lst) > 0):
                country_lst = [item.lower() for item in country_lst]
                for j in range(0, len(country_lst)):
                    if (srt.lower() in country_lst or end.lower() in country_lst):
                        srt_ = self.country_names.index(srt)
                        end_ = self.country_names.index(end)
                        Xb.append(self.country_lat_long[srt_][0]) 
                        Xb.append(self.country_lat_long[end_][0]) 
                        Xb.append(None)
                        Yb.append(self.country_lat_long[srt_][1])
                        Yb.append(self.country_lat_long[end_][1])
                        Yb.append(None)
            srt = self.country_names.index(srt)
            end = self.country_names.index(end)
            Xa.append(self.country_lat_long[srt][0]) 
            Xa.append(self.country_lat_long[end][0]) 
            Xa.append(None)
            Ya.append(self.country_lat_long[srt][1])
            Ya.append(self.country_lat_long[end][1])
            Ya.append(None) 
        data   = dict(type                  = 'choropleth',
                      locations             = iso_3,
                      locationmode          = 'ISO-3',
                      colorscale            = 'sunsetdark', 
                      z                     = vals,
                      hoverinfo             = 'none'
                      )
        edges  = go.Scattergeo(lat          = Xa,
                               lon          = Ya,
                               mode         = 'lines',
                               line         = dict(color = 'rgba(15, 84, 26, 0.25)', width = 1, dash = 'solid'),
                               hoverinfo    = 'none',
                               name         = ''
                               )
        edge_h = go.Scattergeo(lat          = Xb,
                               lon          = Yb,
                               mode         = 'lines',
                               line         = dict(color = 'rgba(255, 3, 45, 0.85)', width = 1, dash = 'solid'),
                               hoverinfo    = 'none',
                               name         = ''
                               )
        nodes  = go.Scattergeo(lon          = lon_,
                               lat          = lat_,
                               text         = text,
                               textfont     = dict(color = 'black', family =  'Times New Roman', size = 10),
                               textposition = 'top center',
                               mode         = 'markers+text',
                               marker       = dict(size = 7, color = 'white', line_color = 'black', line_width = 1),
                               hoverinfo    = 'text',
                               hovertext    = nids_list,
                               name         = '',
                               )
        layout = dict(geo = {'scope': 'world'}, showlegend = False, hovermode  = 'closest',  hoverlabel = dict(font_size = 12), margin = dict(b = 10, l = 5, r = 5, t = 10))
        if (connections == True):
            geo_data = [data, edges]
        else:
            geo_data = [data]
        if (len(country_lst) > 0):
           geo_data.append(edge_h)
        geo_data.append(nodes)
        fig_cm = go.Figure(data = geo_data, layout = layout)
        fig_cm.update_geos(resolution     = 50,
                        showcoastlines = True,  coastlinecolor = 'black',
                        showland       = True,  landcolor      = '#f0f0f0',
                        showocean      = True,  oceancolor     = '#7fcdff',  # '#def3f6', '#7fcdff',
                        showlakes      = False, lakecolor      = 'blue',
                        showrivers     = False, rivercolor     = 'blue',
            			lataxis        = dict(  range          = [-60, 90]), # clip Antarctica
                        )   
        fig_cm.show()
        return

    # Function: Direct Network from Adjacency Matrix
    def network_adj_dir(self, view = 'browser', min_count = 1, node_size = -1, node_labels = False, local_nodes = False):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True and node_size == -1):
            mode = 'markers+text'
            size = 50
        elif (node_labels == False and node_size == -1):
            mode = 'markers'
            size = 10
        elif (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        self.__adjacency_matrix_ref(min_count, local_nodes)
        adjacency_matrix = self.matrix_r.values
        G                = nx.DiGraph()
        rows, cols       = np.where(adjacency_matrix >= 1)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        u_rows           = list(set(rows.tolist()))
        u_cols           = list(set(cols.tolist()))
        labels           = [self.labels_r[item] for item in u_cols]
        labels           = sorted(labels, key = self.natsort)
        for name in labels: 
            if (name.find('r_') != -1):
                color = 'red'
                year  = self.dy_ref[ int(name.replace('r_','')) ]
                if (len(self.u_ref) > 0):
                    n_id  = self.u_ref [ int(name.replace('r_','')) ]
                else:
                    n_id  = ''
                G.add_node(name, color = color,  year = year, n_id = n_id)
            else:
                if (int(name.replace('r_','')) not in u_rows):
                    u_rows.append(int(name.replace('r_','')))
        u_rows = [str(item) for item in u_rows]
        u_rows = sorted(u_rows, key = self.natsort)
        for name in u_rows:
            color = 'blue'
            year  = int(self.dy[ int(name) ])
            n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
            G.add_node(name, color = color, year = year, n_id = n_id )
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = str(srt)
            end_     = self.labels_r[end]
            if ( end_ != '-1' ):
                G.add_edge(srt_, end_)
        color          = [G.nodes[n]['color'] if len(G.nodes[n]) > 0 else 'black' for n in G.nodes()]
        self.pos       = nx.circular_layout(G)
        self.node_list = list(G.nodes)
        self.edge_list = list(G.edges)
        self.nids_list = [G.nodes[n]['n_id'] for n in G.nodes()]
        self.nids_list = ['<br>'.join(textwrap.wrap(txt, width = 50)) for txt in self.nids_list]
        self.nids_list = ['id: '+self.node_list[i]+'<br>'+self.nids_list[i] for i in range(0, len(self.nids_list))]
        self.Xn        = [self.pos[k][0] for k in self.node_list]
        self.Yn        = [self.pos[k][1] for k in self.node_list]
        Xa             = []
        Ya             = []
        for edge in self.edge_list:
            Xa.append(self.pos[edge[0]][0]*0.97)
            Xa.append(self.pos[edge[1]][0]*0.97)
            Xa.append(None)
            Ya.append(self.pos[edge[0]][1]*0.97)
            Ya.append(self.pos[edge[1]][1]*0.97)
            Ya.append(None)
        a_trace = go.Scatter(x         = Xa,
                             y         = Ya,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'dash'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = self.Xn,
                             y         = self.Yn,
                             opacity   = 0.45,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = self.node_list,
                             hoverinfo = 'text',
                             hovertext = self.nids_list,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        self.fig = go.Figure(data = [n_trace, a_trace], layout = layout)
        self.fig.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        self.fig.update_traces(textfont_size = 10, textfont_color = 'yellow') 
        self.fig.show()
        return

    # Function: Network from Adjacency Matrix 
    def network_adj(self, view = 'browser', adj_type = 'aut', min_count = 2, node_size = -1, node_labels = False, label_type = 'id', centrality = None): 
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True):
            mode = 'markers+text'
            size = 17
        else:
            mode = 'markers'
            size = 10
        if (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        if   (adj_type == 'aut'):
            self.__adjacency_matrix_aut(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_aut
        elif (adj_type == 'cout'):
            self.__adjacency_matrix_ctr(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_ctr
        elif (adj_type == 'inst'):
            self.__adjacency_matrix_inst(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_uni
        elif (adj_type == 'kwa'):
            self.__adjacency_matrix_kwa(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_kwa
        elif (adj_type == 'kwp'):
            self.__adjacency_matrix_kwp(min_count)
            adjacency_matrix = self.matrix_a.values
            dict_            = self.dict_id_kwp
        rows, cols = np.where(adjacency_matrix >= 1)
        edges      = list(zip(rows.tolist(), cols.tolist()))
        u_cols     = list(set(cols.tolist()))
        self.H     = nx.Graph()
        if (adj_type == 'aut'):
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[int(name.replace('a_',''))]
                n_doc = self.doc_aut[int(name.replace('a_',''))]
                n_lhi = self.aut_h[int(name.replace('a_',''))]
                n_id  = self.u_aut[int(name.replace('a_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_doc = n_doc, n_lhi = n_lhi, n_id = n_id )
        elif (adj_type == 'cout'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[int(name.replace('c_',''))]
                n_id  = self.u_ctr[int(name.replace('c_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'inst'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[int(name.replace('i_',''))]
                n_id  = self.u_uni[int(name.replace('i_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'kwa'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[int(name.replace('k_',''))]
                n_id  = self.u_auk[int(name.replace('k_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        elif (adj_type == 'kwp'):   
            for i in range(0, len(u_cols)): 
                name  = self.labels_a[u_cols[i]]
                n_cls = -1
                color = 'white'
                n_coa = self.n_colab[int(name.replace('p_',''))]
                n_id  = self.u_kid[int(name.replace('p_',''))]
                self.H.add_node(name, n_cls = n_cls, color = color, n_coa = n_coa, n_id = n_id )  
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = self.labels_a[srt]
            end_     = self.labels_a[end]
            if ( end_ != '-1'):
                self.H.add_edge(srt_, end_)
        if   (centrality == 'degree'): 
            value            = nx.algorithms.centrality.degree_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Degree'])
            self.table_centr = self.table_centr.sort_values('Degree', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'load'):
            value            = nx.algorithms.centrality.load_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Load'])
            self.table_centr = self.table_centr.sort_values('Load', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'betw'):
            value            = nx.algorithms.centrality.betweenness_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Betweenness'])
            self.table_centr = self.table_centr.sort_values('Betweenness', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'close'):
            value            = nx.algorithms.centrality.closeness_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Closeness'])
            self.table_centr = self.table_centr.sort_values('Closeness', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'eigen'):
            value            = nx.algorithms.centrality.eigenvector_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Eigenvector'])
            self.table_centr = self.table_centr.sort_values('Eigenvector', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'katz'):
            value            = nx.algorithms.centrality.katz_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Katz'])
            self.table_centr = self.table_centr.sort_values('Katz', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        elif (centrality == 'harmonic'):
            value            = nx.algorithms.centrality.harmonic_centrality(self.H)
            color            = [value[n] for n in self.H.nodes()]
            self.table_centr = pd.DataFrame(value.items(), columns = ['Node', 'Harmonic'])
            self.table_centr = self.table_centr.sort_values('Harmonic', ascending = False)
            self.table_centr.insert(0, 'Name', [dict_[self.table_centr.iloc[i, 0]] for i in range(0, self.table_centr.shape[0])])
        else:
            generator        = nx.algorithms.community.girvan_newman(self.H)
            community        = next(generator)
            community_list   = sorted(map(sorted, community))
            for com in community_list:
                community_list.index(com)
                for node in com:
                    self.H.nodes[node]['color'] = self.color_names[community_list.index(com)]
                    self.H.nodes[node]['n_cls'] = community_list.index(com)
            color = [self.H.nodes[n]['color'] for n in self.H.nodes()]
        self.pos_a       = nx.spring_layout(self.H, seed = 42, scale = 1000)
        self.node_list_a = list(self.H.nodes)
        self.edge_list_a = list(self.H.edges)
        if (adj_type == 'aut'):
            docs_list_a      = [self.H.nodes[n]['n_doc'] for n in self.H.nodes()]
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            lhid_list_a      = [self.H.nodes[n]['n_lhi'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                       ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '   +str(clst_list_a[i])+'<br>'+
                                'author:                '    +self.nids_list_a[i].upper()+'<br>'+
                                'documents:         '        +str(docs_list_a[i])+'<br>'+
                                'collaborators:      '       +str(auts_list_a[i])+'<br>'+
                                'local h-index:       '      +str(lhid_list_a[i]) 
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'cout'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'country:               '     +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'inst'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'institution:            '    +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'kwa'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'author keyword:    '         +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        elif (adj_type == 'kwp'):  
            auts_list_a      = [self.H.nodes[n]['n_coa'] for n in self.H.nodes()]
            clst_list_a      = [self.H.nodes[n]['n_cls'] for n in self.H.nodes()]
            self.nids_list_a = [self.H.nodes[n]['n_id']  for n in self.H.nodes()]
            self.nids_list_a = ['id:                        ' +self.node_list_a[i]+'<br>'+
                                'cluster:                '    +str(clst_list_a[i])+'<br>'+
                                'keyword plus:     '          +self.nids_list_a[i].upper()+'<br>' +
                                'collaborators:      '        +str(auts_list_a[i])
                                for i in range(0, len(self.nids_list_a))]
        self.Xv = [self.pos_a[k][0] for k in self.node_list_a]
        self.Yv = [self.pos_a[k][1] for k in self.node_list_a]
        Xe      = []
        Ye      = []
        if (label_type != 'id'):
            if (adj_type == 'aut'):
                self.node_list_a = [ self.dict_id_aut[item] for item in self.node_list_a]
            elif (adj_type == 'cout'):
                self.node_list_a = [ self.dict_id_ctr[item] for item in self.node_list_a]
            elif (adj_type == 'inst'): 
                self.node_list_a = [ self.dict_id_uni[item] for item in self.node_list_a]
            elif (adj_type == 'kwa'):
                self.node_list_a = [ self.dict_id_kwa[item] for item in self.node_list_a]
            elif (adj_type == 'kwp'): 
                self.node_list_a = [ self.dict_id_kwp[item] for item in self.node_list_a]
        for edge in self.edge_list_a:
            Xe.append(self.pos_a[edge[0]][0]*1.00)
            Xe.append(self.pos_a[edge[1]][0]*1.00)
            Xe.append(None)
            Ye.append(self.pos_a[edge[0]][1]*1.00)
            Ye.append(self.pos_a[edge[1]][1]*1.00)
            Ye.append(None)
        a_trace = go.Scatter(x         = Xe,
                             y         = Ye,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        n_trace = go.Scatter(x         = self.Xv,
                             y         = self.Yv,
                             opacity   = 0.57,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = color, line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = self.node_list_a,
                             hoverinfo = 'text',
                             hovertext = self.nids_list_a,
                             name      = ''
                             )
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = False),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            )
        self.fig_a = go.Figure(data = [n_trace, a_trace], layout = layout)
        self.fig_a.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        self.fig_a.update_traces(textfont_size = 10, textfont_color = 'blue', textposition = 'top center') 
        self.fig_a.show()
        if (label_type != 'id'):
            if (adj_type == 'aut'):
                self.node_list_a = [ self.dict_aut_id[item] for item in self.node_list_a]
            elif (adj_type == 'cout'):
                self.node_list_a = [ self.dict_ctr_id[item] for item in self.node_list_a]
            elif (adj_type == 'inst'): 
                self.node_list_a = [ self.dict_uni_id[item] for item in self.node_list_a]
            elif (adj_type == 'kwa'):
                self.node_list_a = [ self.dict_kwa_id[item] for item in self.node_list_a]
            elif (adj_type == 'kwp'): 
                self.node_list_a = [ self.dict_kwp_id[item] for item in self.node_list_a]
        return

    # Function: Find Connected Nodes from Direct Network
    def find_nodes_dir(self, view = 'browser', article_ids = [], ref_ids = [], node_size = -1):
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_size > 0):
            size = node_size
        else:
            size = 50
        fig_ = go.Figure(self.fig)
        if (len(article_ids) > 0 or len(ref_ids) > 0):
            if (len(article_ids) > 0):
                edge_list_ai = []
                idx_ids      = []
                color_ids    = []
                text_ids     = []
                hover_ids    = []
                article_ids  = [str(int(item)) for item in article_ids]
                idx_ids.extend([self.node_list.index(node) for node in article_ids])
                color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in article_ids])
                text_ids.extend([node for node in article_ids])
                hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in article_ids])
                for ids in article_ids:
                    edge_list_ai.extend([item for item in self.edge_list if ids == item[0]])
                    node_pair = [self.edge_list[i][1] for i in range(0, len(self.edge_list)) if ids in self.edge_list[i]]
                    idx_ids.extend([self.node_list.index(node) for node in node_pair])
                    color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in node_pair])
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in node_pair])
                xa = []
                ya = []
                if (len(edge_list_ai) > 0):
                    for edge in edge_list_ai:
                        xa.append(self.pos[edge[0]][0]*0.97)
                        xa.append(self.pos[edge[1]][0]*0.97)
                        ya.append(self.pos[edge[0]][1]*0.97)
                        ya.append(self.pos[edge[1]][1]*0.97)
                    for i in range(0, len(xa), 2):
                        fig_.add_annotation(
                                           x          = xa[i + 1],  # to x
                                           y          = ya[i + 1],  # to y
                                           ax         = xa[i + 0],  # from x
                                           ay         = ya[i + 0],  # from y
                                           xref       = 'x',
                                           yref       = 'y',
                                           axref      = 'x',
                                           ayref      = 'y',
                                           text       = '',
                                           showarrow  = True,
                                           arrowhead  = 3,
                                           arrowsize  = 1.2,
                                           arrowwidth = 1,
                                           arrowcolor = 'black',
                                           opacity    = 0.9
                                           )
                xn = [self.Xn[i] for i in idx_ids]
                yn = [self.Yn[i] for i in idx_ids]
                fig_.add_trace(go.Scatter(x              = xn,
                                          y              = yn,
                                          mode           = 'markers+text',
                                          marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                          text           = text_ids,
                                          hoverinfo      = 'text',
                                          hovertext      = hover_ids,
                                          textfont_size  = 10, 
                                          textfont_color = 'yellow',
                                          name           = ''
                                          ))
            if (len(ref_ids) > 0):
                edge_list_ri = []
                idx_ids      = []
                color_ids    = []
                text_ids     = []
                hover_ids    = []
                ref_ids      = [str(item) for item in ref_ids]
                idx_ids.extend([self.node_list.index(node) for node in ref_ids])
                color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in ref_ids])
                text_ids.extend([node for node in ref_ids])
                hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in ref_ids])
                for ids in ref_ids:
                    edge_list_ri.extend([item for item in self.edge_list if ids == item[1]]) 
                    node_pair = [self.edge_list[i][0] for i in range(0, len(self.edge_list)) if ids in self.edge_list[i]]
                    idx_ids.extend([self.node_list.index(node) for node in node_pair])
                    color_ids.extend(['red' if node.find('r_') >= 0 else 'blue' for node in node_pair])
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list[self.node_list.index(node)] for node in node_pair])
                xa = []
                ya = []
                if (len(edge_list_ri) > 0):
                    for edge in edge_list_ri:
                        xa.append(self.pos[edge[0]][0]*0.97)
                        xa.append(self.pos[edge[1]][0]*0.97)
                        ya.append(self.pos[edge[0]][1]*0.97)
                        ya.append(self.pos[edge[1]][1]*0.97)
                    for i in range(0, len(xa), 2):
                        fig_.add_annotation(
                                           x          = xa[i + 1],  # to x
                                           y          = ya[i + 1],  # to y
                                           ax         = xa[i + 0],  # from x
                                           ay         = ya[i + 0],  # from y
                                           xref       = 'x',
                                           yref       = 'y',
                                           axref      = 'x',
                                           ayref      = 'y',
                                           text       = '',
                                           showarrow  = True,
                                           arrowhead  = 3,
                                           arrowsize  = 1.2,
                                           arrowwidth = 1,
                                           arrowcolor = 'black',
                                           opacity    = 0.9
                                           )
                xn = [self.Xn[i] for i in idx_ids]
                yn = [self.Yn[i] for i in idx_ids]
                fig_.add_trace(go.Scatter(x              = xn,
                                          y              = yn,
                                          mode           = 'markers+text',
                                          marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                          text           = text_ids,
                                          hoverinfo      = 'text',
                                          hovertext      = hover_ids,
                                          textfont_size  = 10, 
                                          textfont_color = 'yellow',
                                          name           = ''
                                          ))
        fig_.show()
        return 

    # Function: Find Connected Nodes
    def find_nodes(self, node_ids = [], node_name = [], node_size = -1, node_only = False):
        flag = False
        if (len(node_ids) == 0 and len(node_name) > 0):
            if   (node_name[0] in self.dict_aut_id.keys()):
                node_ids = [self.dict_aut_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_ctr_id.keys()):
                node_ids = [self.dict_ctr_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_uni_id.keys()):
                node_ids = [self.dict_uni_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_kwa_id.keys()):
                node_ids = [self.dict_kwa_id[item] for item in node_name]
                flag     = True
            elif (node_name[0] in self.dict_kwp_id.keys()):
                node_ids = [self.dict_kwp_id[item] for item in node_name]
                flag     = True
        if (node_size > 0):
            size = node_size
        else:
            size = 17
        fig_ = go.Figure(self.fig_a)
        fig_.update_traces(mode = 'markers', line = dict(width = 0), marker = dict(color = 'rgba(0, 0, 0, 0.1)', size = 7)) 
        if (len(node_ids) > 0):
            edge_list_ai = []
            idx_ids      = []
            color_ids    = []
            text_ids     = []
            hover_ids    = []
            idx_ids.extend([self.node_list_a.index(node) for node in node_ids])
            color_ids.extend(['black' for n in node_ids]) # self.H.nodes[n]['color']
            text_ids.extend([node for node in node_ids])
            hover_ids.extend([self.nids_list_a[self.node_list_a.index(node)] for node in node_ids])
            if (node_only != True):
                for ids in node_ids:
                    edge_list_ai.extend([item for item in self.edge_list_a if ids in item and item not in edge_list_ai])
                    node_pair = [self.edge_list_a[i][1] for i in range(0, len(self.edge_list_a)) if ids in self.edge_list_a[i] and self.edge_list_a[i][1] != ids]
                    node_pair.extend([self.edge_list_a[i][0] for i in range(0, len(self.edge_list_a)) if ids in self.edge_list_a[i] and self.edge_list_a[i][0] != ids])
                    idx_ids.extend([self.node_list_a.index(node) for node in node_pair])
                    color_ids.extend(['#e0cc92' for n in node_pair]) # self.H.nodes[n]['color']
                    text_ids.extend([node for node in node_pair])
                    hover_ids.extend([self.nids_list_a[self.node_list_a.index(node)] for node in node_pair])
                xa = []
                ya = []
                for edge in edge_list_ai:
                    xa.append(self.pos_a[edge[0]][0]*1.00)
                    xa.append(self.pos_a[edge[1]][0]*1.00)
                    xa.append(None)
                    ya.append(self.pos_a[edge[0]][1]*1.00)
                    ya.append(self.pos_a[edge[1]][1]*1.00)
                    ya.append(None)
                for i in range(0, len(xa), 2):
                    fig_.add_trace(go.Scatter(x         = xa,
                                              y         = ya,
                                              mode      = 'lines',
                                              line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'solid'),
                                              hoverinfo = 'none',
                                              name      = ''
                                              ))
            xn = [self.Xv[i] for i in idx_ids]
            yn = [self.Yv[i] for i in idx_ids]
            if (flag == True):
                if   (node_name[0] in self.dict_aut_id.keys()):
                    text_ids = [self.dict_id_aut[item] for item in text_ids]
                elif (node_name[0] in self.dict_ctr_id.keys()):
                    text_ids = [self.dict_id_ctr[item] for item in text_ids]
                elif (node_name[0] in self.dict_uni_id.keys()):
                    text_ids = [self.dict_id_uni[item] for item in text_ids]
                elif (node_name[0] in self.dict_kwa_id.keys()):
                    text_ids = [self.dict_id_kwa[item] for item in text_ids]
                elif (node_name[0] in self.dict_kwp_id.keys()):
                    text_ids = [self.dict_id_kwp[item] for item in text_ids]
            fig_.add_trace(go.Scatter(x              = xn,
                                      y              = yn,
                                      mode           = 'markers+text',
                                      marker         = dict(symbol = 'circle-dot', size = size, color = color_ids),
                                      text           = text_ids,
                                      hoverinfo      = 'text',
                                      hovertext      = hover_ids,
                                      textfont_size  = 10, 
                                      name           = ''
                                      ))
        fig_.update_traces(textposition = 'top center') 
        fig_.show()
        return
    
    # Function: Citation History Network
    def network_hist(self, view = 'browser', min_count = 1, node_size = -1, node_labels = False, back = [], forward = []):
        years = list(range(self.date_str, self.date_end+1)) 
        if (view == 'browser' ):
            pio.renderers.default = 'browser'
        if (node_labels == True and node_size == -1):
            mode = 'markers+text'
            size = 50
        elif (node_labels == False and node_size == -1):
            mode = 'markers'
            size = 10
        elif (node_labels == True and node_size > 0):
            mode = 'markers+text'
            size = node_size
        elif (node_labels == False and node_size > 0):
            mode = 'markers'
            size = node_size
        self.__adjacency_matrix_ref(min_count, True)
        adjacency_matrix = self.matrix_r.values
        G                = nx.DiGraph()
        rows, cols       = np.where(adjacency_matrix >= 1)
        edges            = list(zip(rows.tolist(), cols.tolist()))
        u_rows           = list(set(rows.tolist()))
        u_cols           = list(set(cols.tolist()))
        labels           = [self.labels_r[item] for item in u_cols]
        labels           = sorted(labels, key = self.natsort)
        Xn               = []
        Yn               = []
        Xa               = []
        Ya               = []
        ys               = list(range(self.date_str, self.date_end+1))
        dict_y           = dict(zip(ys, list(range(0, len(ys)))))
        flag             = 0
        y_lst            = []
        for name in labels: 
            if (name.find('r_') != -1):
                color = 'red'
                year  = self.dy_ref[ int(name.replace('r_','')) ]
                if (len(self.u_ref) > 0):
                    n_id  = self.u_ref [ int(name.replace('r_','')) ]
                else:
                    n_id  = ''
                if ( year not in y_lst):
                    y_lst.append(year)
                    flag = 0
                elif (year in y_lst):
                    counter = y_lst.count(year)
                    flag    = counter*1.2
                    y_lst.append(year)
                G.add_node(name, color = color,  year = year, n_id = n_id, flag = flag)
            else:
                if (int(name.replace('r_','')) not in u_rows):
                    u_rows.append(int(name.replace('r_','')))
        u_rows = [str(item) for item in u_rows]
        u_rows = sorted(u_rows, key = self.natsort)
        flag   = 0
        y_lst  = []
        for name in u_rows:
            color = 'blue'
            year  = int(self.dy[ int(name) ])
            n_id  = self.data.loc[int(name), 'author']+' ('+self.data.loc[int(name), 'year']+'). '+self.data.loc[int(name), 'title']+'. '+self.data.loc[int(name), 'journal']+'. doi:'+self.data.loc[int(name), 'doi']+'. '
            if ( year not in y_lst):
                y_lst.append(year)
                flag = 0
            elif (year in y_lst):
                counter = y_lst.count(year)
                flag    = counter*1.2
                y_lst.append(year)
            G.add_node(name, color = color, year = year, n_id = n_id, flag = flag)
            Xn.append(dict_y[year])
            Yn.append(flag)
        for i in range(0, len(edges)):
            srt, end = edges[i]
            srt_     = str(srt)
            end_     = self.labels_r[end]
            if ( end_ != '-1' ):
                G.add_edge(srt_, end_)
        node_list = list(G.nodes)
        edge_list = list(G.edges)
        nids_list = [G.nodes[n]['n_id'] for n in G.nodes()]
        nids_list = ['<br>'.join(textwrap.wrap(txt, width = 50)) for txt in nids_list]
        nids_list = ['id: '+node_list[i]+'<br>'+nids_list[i] for i in range(0, len(nids_list))]
        for edge in edge_list:
            Xa.append(dict_y[G.nodes[edge[0]]['year']]) 
            Xa.append(dict_y[G.nodes[edge[1]]['year']])
            Xa.append(None)
            Ya.append(G.nodes[edge[0]]['flag'])
            Ya.append(G.nodes[edge[1]]['flag'])
            Ya.append(None)
        data    = []
        a_trace = go.Scatter(x         = Xa,
                             y         = Ya,
                             mode      = 'lines',
                             line      = dict(color = 'rgba(0, 0, 0, 0.25)', width = 0.5, dash = 'dot'),
                             hoverinfo = 'none',
                             name      = ''
                             )
        data.append(a_trace)
        n_trace = go.Scatter(x         = Xn,
                             y         = Yn,
                             opacity   = 0.45,
                             mode      = mode,
                             marker    = dict(symbol = 'circle-dot', size = size, color = 'blue', line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                             text      = node_list,
                             hoverinfo = 'text',
                             hovertext = nids_list,
                             name      = ''
                             )
        data.append(n_trace)
        layout  = go.Layout(showlegend = False,
                            hovermode  = 'closest',
                            margin     = dict(b = 10, l = 5, r = 5, t = 10),
                            xaxis      = dict(showgrid = False, zeroline = False, showticklabels = True, tickmode = 'array', tickvals       = list(range(0, len(years))), ticktext = years, tickangle =  90),
                            yaxis      = dict(showgrid = False, zeroline = False, showticklabels = False)
                            ) 
        if (len(back) > 0):
            e_lst = []
            Xb    = []
            Yb    = []
            Xm    = []
            Ym    = []
            n_lst = []
            t_lst = []
            for item in back:
                item = str(item)
                for edge in edge_list:
                    a, b = edge
                    if (item == a):
                        e_lst.append(edge)
                        back.append(b)
            for edge in e_lst:
                Xb.append(dict_y[G.nodes[edge[0]]['year']]) 
                Xb.append(dict_y[G.nodes[edge[1]]['year']])
                Xb.append(None)
                Yb.append(G.nodes[edge[0]]['flag'])
                Yb.append(G.nodes[edge[1]]['flag'])
                Yb.append(None)
                Xm.append(dict_y[G.nodes[edge[0]]['year']])
                Ym.append(G.nodes[edge[0]]['flag'])
                Xm.append(dict_y[G.nodes[edge[1]]['year']])
                Ym.append(G.nodes[edge[1]]['flag'])
                n_lst.append(edge[0])
                n_lst.append(edge[1])
                t_lst.append('id: '+(edge[0]+'<br>'+'<br>'.join(textwrap.wrap(G.nodes[edge[0]]['n_id'], width = 50))))
                t_lst.append('id: '+(edge[1]+'<br>'+'<br>'.join(textwrap.wrap(G.nodes[edge[1]]['n_id'], width = 50))))
            b_trace = go.Scatter(x         = Xb,
                                 y         = Yb,
                                 mode      = 'lines',
                                 line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1, dash = 'solid'),
                                 hoverinfo = 'none',
                                 name      = ''
                                 )
            data.append(b_trace)
            m_trace = go.Scatter(x         = Xm,
                                 y         = Ym,
                                 opacity   = 1,
                                 mode      = 'markers+text',
                                 marker    = dict(symbol = 'circle-dot', size = 50, color = 'blue', line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                                 text      = n_lst,
                                 hoverinfo = 'text',
                                 hovertext = t_lst,
                                 name      = ''
                                 )
            data.append(m_trace)
        if (len(forward) > 0):
            e_lst = []
            Xb    = []
            Yb    = []
            Xm    = []
            Ym    = []
            n_lst = []
            t_lst = []
            for item in forward:
                item = str(item)
                for edge in edge_list:
                    a, b = edge
                    if (item == b):
                        e_lst.append(edge)
                        forward.append(a)
            for edge in e_lst:
                Xb.append(dict_y[G.nodes[edge[0]]['year']]) 
                Xb.append(dict_y[G.nodes[edge[1]]['year']])
                Xb.append(None)
                Yb.append(G.nodes[edge[0]]['flag'])
                Yb.append(G.nodes[edge[1]]['flag'])
                Yb.append(None)
                Xm.append(dict_y[G.nodes[edge[0]]['year']])
                Ym.append(G.nodes[edge[0]]['flag'])
                Xm.append(dict_y[G.nodes[edge[1]]['year']])
                Ym.append(G.nodes[edge[1]]['flag'])
                n_lst.append(edge[0])
                n_lst.append(edge[1])
                t_lst.append('id: '+(edge[0]+'<br>'+'<br>'.join(textwrap.wrap(G.nodes[edge[0]]['n_id'], width = 50))))
                t_lst.append('id: '+(edge[1]+'<br>'+'<br>'.join(textwrap.wrap(G.nodes[edge[1]]['n_id'], width = 50))))
            c_trace = go.Scatter(x         = Xb,
                                 y         = Yb,
                                 mode      = 'lines',
                                 line      = dict(color = 'rgba(255, 0, 0, 1)', width = 1, dash = 'solid'),
                                 hoverinfo = 'none',
                                 name      = ''
                                 )
            data.append(c_trace)
            p_trace = go.Scatter(x         = Xm,
                                 y         = Ym,
                                 opacity   = 1,
                                 mode      = 'markers+text',
                                 marker    = dict(symbol = 'circle-dot', size = 50, color = 'blue', line = dict(color = 'rgb(50, 50, 50)', width = 0.15)),
                                 text      = n_lst,
                                 hoverinfo = 'text',
                                 hovertext = t_lst,
                                 name      = ''
                                 )
            data.append(p_trace)
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(yaxis = dict(scaleanchor = 'x', scaleratio = 0.5), plot_bgcolor = 'rgb(255, 255, 255)',  hoverlabel = dict(font_size = 12))
        fig.update_traces(textfont_size = 10, textfont_color = 'yellow') 
        fig.show()
        return

############################################################################

    # Function: Sentence Embeddings # 'abs', 'title', 'kwa', 'kwp'
    def create_embeddings(self, stop_words = ['en'], rmv_custom_words = [], corpus_type = 'abs'):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        if  (corpus_type == 'abs'):
            corpus = self.data['abstract']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'title'):
            corpus = self.data['title']
            corpus = corpus.tolist()
            corpus = self.clear_text(corpus, stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words)
        elif (corpus_type == 'kwa'): 
            corpus = self.data['author_keywords']
            corpus = corpus.tolist()
        elif (corpus_type == 'kwp'):
            corpus = self.data['keywords']
            corpus = corpus.tolist()
        self.embds = model.encode(corpus)
        return self  

############################################################################

    # Function: Topics - Create
    def topics_creation(self, stop_words = ['en'], rmv_custom_words = [], embeddings = False):
        umap_model = UMAP(n_neighbors = 15, n_components = 5, min_dist = 0.0, metric = 'cosine', random_state = 1001)
        if (embeddings ==  False):
            self.topic_model = BERTopic(umap_model = umap_model, calculate_probabilities = True)
        else:
            sentence_model   = SentenceTransformer('all-MiniLM-L6-v2')
            self.topic_model = BERTopic(umap_model = umap_model, calculate_probabilities = True, embedding_model = sentence_model)
        self.topic_corpus       = self.clear_text(self.data['abstract'], stop_words = stop_words, lowercase = True, rmv_accents = True, rmv_special_chars = True, rmv_numbers = True, rmv_custom_words = rmv_custom_words, verbose = False)
        self.topics, self.probs = self.topic_model.fit_transform(self.topic_corpus)
        self.topic_info         = self.topic_model.get_topic_info()
        print(self.topic_info)
        return self   

    # Function: Topics - Align Outliers
    #def topics_align_outliers(self):
        #self.topics     = [np.argmax(probs[i,:]) for i in range(0, self.probs.shape[0])]
        #self.topic_info = self.topic_info[self.topic_info.Topic != -1]
        #for i in range(0, self.topic_info.shape[0]):
            #self.topic_info.iloc[i, 1] = self.topics.count(self.topic_info.iloc[i, 0])
        #print(self.topic_info)
        #return self

    # Function: Topics - Main Representatives
    def topics_representatives(self):
        docs        = [[] for _ in range(0, self.topic_info.shape[0])]
        papers      = self.topic_model.get_representative_docs()
        self.df_rep = pd.DataFrame(np.zeros((self.topic_info.shape[0], 2)), columns = ['Topic', 'Docs'])
        for i in range(0, self.topic_info.shape[0]):
            if (self.topic_info.iloc[i, 0] != -1):
                paper = papers[self.topic_info.iloc[i, 0]]
                for item in paper:
                    docs[i].append(self.topic_corpus.index(item))
            self.df_rep.iloc[i, 0] = self.topic_info.iloc[i, 0]
            self.df_rep.iloc[i, 1] = '; '.join(map(str, docs[i]))
        return self.df_rep
        
    # Function: Topics - Reduce
    def topics_reduction(self, topicsn = 3):
        self.topics, self.probs = self.topic_model.reduce_topics(self.topic_corpus, self.topics, self.probs, nr_topics = topicsn - 1)
        self.topic_info         = self.topic_model.get_topic_info()
        print(self.topic_info)
        return self 
    
    # Function: Graph Topics - Topics
    def graph_topics(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = ['Topic ' + str(self.topic_info.iloc[i, 0]) + ' ( Count = ' + str(self.topic_info.iloc[i, 1]) + ') ' for i in range(0, self.topic_info.shape[0])]
        column       = 1
        columns      = 4
        row          = 1
        rows         = int(np.ceil(self.topic_info.shape[0] / columns))
        fig          = ps.make_subplots(rows               = rows,
                                        cols               = columns,
                                        shared_yaxes       = False,
                                        shared_xaxes       = False,
                                        horizontal_spacing = 0.1,
                                        vertical_spacing   = 0.4 / rows if rows > 1 else 0,
                                        subplot_titles     = topics_label
                                        )
        for i in range(0, self.topic_info.shape[0]):
            sequence = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            words    = [str(item[0]) for item in sequence]
            values   = [str(item[1]) for item in sequence]
            trace    = go.Bar(x           = values,
                              y           = words,
                              orientation = 'h',
                              marker      = dict(color = self.color_names[i], line = dict(color = 'black', width = 1))
                              )
            fig.append_trace(trace, row, column)
            if (column == columns):
                column = 1
                row    = row + 1
            else:
                column = column + 1
        fig.update_xaxes(showticklabels = False)
        fig.update_layout(paper_bgcolor = 'rgb(255, 255, 255)', plot_bgcolor = 'rgb(255, 255, 255)', showlegend = False)
        fig.show()
        return self 
    
    # Function: Graph Topics - Topics Distribution
    def graph_topics_distribution(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        topics_count = []
        words        = []
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append('Topic ' + str(self.topic_info.iloc[i, 0]))
            topics_count.append(self.topic_info.iloc[i, 1])
            sequence = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            sequence = ['-'+str(item[0]) for item in sequence]
            words.append('Count: ' + str(self.topic_info.iloc[i, 1]) +'<br>'+'<br>'+ 'Words: ' +'<br>'+ '<br>'.join(sequence))
        fig = go.Figure(go.Bar(x           = topics_label,
                               y           = topics_count,
                               orientation = 'v',
                               hoverinfo   = 'text',
                               hovertext   = words,
                               marker      = dict(color = 'rgba(78, 246, 215, 0.6)', line = dict(color = 'black', width = 1)),
                               name        = ''
                              ),
                        )
        fig.update_xaxes(zeroline = False)
        fig.update_layout(paper_bgcolor = 'rgb(189, 189, 189)', plot_bgcolor = 'rgb(189, 189, 189)')
        fig.show()
        return self 
    
    # Function: Graph Topics - Projected Topics 
    def graph_topics_projection(self, view = 'browser', method = 'tsvd'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        topics_count = []
        words        = []
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append(str(self.topic_info.iloc[i, 0]))
            topics_count.append(self.topic_info.iloc[i, 1])
            sequence  = self.topic_model.get_topic(self.topic_info.iloc[i, 0])
            sequence  = ['-'+str(item[0]) for item in sequence]
            words.append('Count: ' + str(self.topic_info.iloc[i, 1]) +'<br>'+'<br>'+ 'Words: ' +'<br>'+ '<br>'.join(sequence))
        try:
            embeddings = self.topic_model.c_tf_idf.toarray()
        except:
            embeddings = self.topic_model.c_tf_idf_.toarray()
        if (method.lower() == 'umap'):
            decomposition = UMAP(n_components = 2, random_state = 1001)
        else:
            decomposition = tsvd(n_components = 2, random_state = 1001)
        transformed   = decomposition.fit_transform(embeddings)
        fig           = go.Figure(go.Scatter(x           = transformed[:,0],
                                             y           = transformed[:,1],
                                             opacity     = 0.85,
                                             mode        = 'markers+text',
                                             marker      = dict(symbol = 'circle-dot', color = 'rgba(250, 240, 52, 0.75)', line = dict(color = 'black', width = 1)), 
                                             marker_size = topics_count,
                                             text        = topics_label,
                                             hoverinfo   = 'text',
                                             hovertext   = words,
                                             name        = ''
                                             ),
                                  )
        x_range = (transformed[:,0].min() - abs((transformed[:,0].min()) * .35), transformed[:,0].max() + abs((transformed[:,0].max()) * .35))
        y_range = (transformed[:,1].min() - abs((transformed[:,1].min()) * .35), transformed[:,1].max() + abs((transformed[:,1].max()) * .35))
        fig.update_xaxes(range = x_range, showticklabels = False)
        fig.update_yaxes(range = y_range, showticklabels = False)
        fig.add_shape(type = 'line', x0 = sum(x_range)/2, y0 = y_range[0], x1 = sum(x_range)/2, y1 = y_range[1], line = dict(color = 'rgb(0, 0, 0)', width = 0.5))
        fig.add_shape(type = 'line', x0 = x_range[0], y0 = sum(y_range)/2, x1 = x_range[1], y1 = sum(y_range)/2, line = dict(color = 'rgb(0, 0, 0)', width = 0.5))
        fig.add_annotation(x = x_range[0], y = sum(y_range)/2, text = '<b>D1<b>', showarrow = False, yshift = 10)
        fig.add_annotation(y = y_range[1], x = sum(x_range)/2, text = '<b>D2<b>', showarrow = False, xshift = 10)
        fig.update_layout(paper_bgcolor = 'rgb(235, 235, 235)', plot_bgcolor = 'rgb(235, 235, 235)', xaxis = dict(showgrid = False, zeroline = False), yaxis = dict(showgrid = False, zeroline = False))
        fig.show()
        return self 
    
    # Function: Graph Topics - Topics Heatmap
    def graph_topics_heatmap(self, view = 'browser'):
        if (view == 'browser'):
            pio.renderers.default = 'browser'
        topics_label = []
        try:
            embeddings = self.topic_model.c_tf_idf.toarray()
        except:
            embeddings = self.topic_model.c_tf_idf_.toarray()
        dist_matrix  = cosine_similarity(embeddings)
        for i in range(0, self.topic_info.shape[0]):
            topics_label.append('Topic ' + str(self.topic_info.iloc[i, 0]))
        trace = go.Heatmap(z          = dist_matrix,
                           x          = topics_label,
                           y          = topics_label,
                           zmin       = -1,
                           zmax       =  1,
                           xgap       =  1,
                           ygap       =  1,
                           text       = np.around(dist_matrix, decimals = 2),
                           hoverinfo  = 'text',
                           colorscale = 'thermal'
                          )
        layout = go.Layout(title_text = 'Topics Heatmap', xaxis_showgrid = False, yaxis_showgrid = False, yaxis_autorange = 'reversed')
        fig    = go.Figure(data = [trace], layout = layout)
        fig.show()
        return self
 
############################################################################

    # Function: Abstractive Text Summarization # Model Name List = https://huggingface.co/models?pipeline_tag=summarization&sort=downloads&search=pegasus
    def summarize_abst_peg(self, article_ids = [], model_name = 'google/pegasus-xsum'):
        abstracts = self.data['abstract']
        corpus    = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOW' and i in article_ids):
                corpus.append(abstracts.iloc[i])
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', len(corpus))
            print('')
            corpus    = ' '.join(corpus)
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            pegasus   = PegasusForConditionalGeneration.from_pretrained(model_name)
            tokens    = tokenizer(corpus, truncation = True, padding = 'longest', return_tensors = 'pt')
            summary   = pegasus.generate(**tokens) # max_new_tokens = 1024, max_length = 1024, 
            summary   = tokenizer.decode(summary[0])
        else:
            summary   = 'No abstracts were found in the selected set of documents'
        return summary
    
    # Function: Extractive Text Summarization
    def summarize_ext_bert(self, article_ids = []):
        abstracts = self.data['abstract']
        corpus    = []
        if (len(article_ids) == 0):
            article_ids = [i for i in range(0, abstracts.shape[0])]
        else:
            article_ids = [int(item) for item in article_ids]
        for i in range(0, abstracts.shape[0]):
            if (abstracts.iloc[i] != 'UNKNOW' and i in article_ids):
                corpus.append(abstracts.iloc[i])
        if (len(corpus) > 0):
            print('')
            print('Total Number of Valid Abstracts: ', len(corpus))
            print('')
            corpus     = ' '.join(corpus)
            bert_model = Summarizer()
            summary    = ''.join(bert_model(corpus, min_length = 5))
        else:
            summary    = 'No abstracts were found in the selected set of documents'
        return summary

############################################################################
