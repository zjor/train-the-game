import torch
import numpy as np

data = [(0.10610737385376351, 0, 0.5, 0), (0.10610737385376351, 0, 0.5, 0), (0.0632274879378179, 0, 0.5, 0), (0.0632274879378179, 0, 0.5, 0), (0.021887472761503546, 0, 0.5, 0), (0.021887472761503546, 0, 0.5, 0), (-0.01763839792221711, 0, 0.5, 0), (-0.01763839792221711, 0, 0.5, 0), (-0.055095225963569194, 0, 0.5, 0), (-0.055095225963569194, 0, 0.5, 0), (-0.09024962185344959, 0, 0.5, 0), (-0.09024962185344959, 0, 0.5, 0), (-0.12289162393160086, 0, 0.5, 0), (-0.12289162393160086, 0, 0.5, 0), (-0.1528363851333857, 0, 0.5, 0), (-0.1528363851333857, 0, 0.5, 0), (-0.17992561105990224, 0, 0.5, 0), (-0.17992561105990224, 0, 0.5, 0), (-0.20402873574205482, 0, 0.5, 0), (-0.20402873574205482, 0, 0.5, 0), (-0.2250438241963672, 0, 0.5, 0), (-0.2250438241963672, 1, 0.499672767379301, -1), (-0.2428981937096012, 2, 0.49901840181612916, -1), (-0.25754874871017247, 3, 0.4980371026365741, -1), (-0.25754874871017247, 4, 0.4967291687538718, -1), (-0.2689820270556183, 5, 0.4950949985773533, -1), (-0.2689820270556183, 6, 0.49313508989108484, -1), (-0.2772139585551821, 7, 0.4908500397022383, -1), (-0.2822893395228727, 8, 0.48824054405923706, -1), (-0.2822893395228727, 9, 0.48530739783973276, -1), (-0.28428103008731975, 10, 0.48205149450847784, -1), (-0.28428103008731975, 11, 0.47847382584516773, -1), (-0.28328888383902207, 12, 0.47457548164233465, -1), (-0.2794384221426837, 12, 0.47067713743950157, 0), (-0.2794384221426837, 12, 0.4667787932366685, 0), (-0.2728792680529539, 12, 0.4628804490338354, 0), (-0.2728792680529539, 12, 0.4589821048310024, 0), (-0.26378335721821006, 12, 0.4550837606281693, 0), (-0.26378335721821006, 12, 0.4511854164253362, 0), (-0.2523429454130491, 12, 0.44728707222250313, 0), (-0.2523429454130491, 12, 0.44338872801967005, 0), (-0.23876843438194806, 12, 0.43949038381683697, 0), (-0.23876843438194806, 12, 0.43559203961400395, 0), (-0.22328603948253067, 12, 0.43169369541117086, 0), (-0.22328603948253067, 12, 0.4277953512083378, 0), (-0.206135324168039, 12, 0.4238970070055047, 0), (-0.206135324168039, 13, 0.41967917473655725, -1), (-0.18756662762869467, 14, 0.4151431391940634, -1), (-0.18756662762869467, 15, 0.4102902820983912, -1), (-0.16783841290740212, 16, 0.4051220816768224, -1), (-0.16783841290740212, 17, 0.3996401122132711, -1), (-0.1472145635064825, 18, 0.39384604356874076, -1), (-0.1472145635064825, 18, 0.38805197492421045, 0), (-0.12596165690188116, 18, 0.3822579062796801, 0), (-0.12596165690188116, 18, 0.37646383763514973, 0), (-0.10434624347584726, 18, 0.37066976899061943, 0), (-0.10434624347584726, 18, 0.36487570034608907, 0), (-0.08263215916807928, 18, 0.3590816317015587, 0), (-0.08263215916807928, 18, 0.3532875630570284, 0), (-0.0610778996316856, 18, 0.34749349441249805, 0), (-0.0610778996316856, 17, 0.34201152494894677, 1), (-0.03993408287025847, 16, 0.336843324527378, 1), (-0.03993408287025847, 15, 0.3319904674317058, 1), (-0.03993408287025847, 14, 0.3274544318892119, 1), (-0.01944102623534081, 13, 0.32323659962026446, 1), (-0.01944102623534081, 12, 0.31933825541743144, 1), (-0.01944102623534081, 11, 0.3157605867541212, 1), (0.00017353770790799403, 10, 0.3125046834228663, 1), (0.00017353770790799403, 9, 0.309571537203362, 1), (0.018696583569413956, 8, 0.3069620415603607, 1), (0.018696583569413956, 7, 0.3046769913715142, 1), (0.018696583569413956, 6, 0.30271708268524566, 1), (0.03593173277083535, 5, 0.30108291250872704, 1), (0.03593173277083535, 5, 0.2994487423322084, 0), (0.051700991514451684, 5, 0.2978145721556898, 0), (0.051700991514451684, 5, 0.29618040197917117, 0), (0.06584627045578134, 5, 0.2945462318026525, 0), (0.06584627045578134, 5, 0.29291206162613387, 0), (0.07823067190967979, 5, 0.29127789144961524, 0), (0.07823067190967979, 5, 0.2896437212730966, 0), (0.088739533021843, 4, 0.2883357873903942, 1), (0.088739533021843, 3, 0.28735448821083903, 1), (0.09728121605497361, 2, 0.2867001226476671, 1), (0.09728121605497361, 1, 0.286372890026968, 1), (0.09728121605497361, 0, 0.286372890026968, 1), (0.10378763974143937, -1, 0.2867001226476671, 1), (0.10821454851122159, -2, 0.28735448821083903, 1), (0.10821454851122159, -3, 0.2883357873903942, 1), (0.1105415192839857, -4, 0.2896437212730966, 1), (0.1105415192839857, -4, 0.290951655155799, 0), (0.11077170838558245, -4, 0.29225958903850136, 0), (0.11077170838558245, -4, 0.29356752292120375, 0), (0.10893134398082915, -4, 0.29487545680390614, 0), (0.10893134398082915, -4, 0.29618339068660854, 0), (0.10506897217506211, -4, 0.2974913245693109, 0), (0.10506897217506211, -4, 0.29879925845201327, 0), (0.09925446759657797, -4, 0.30010719233471567, 0), (0.09925446759657797, -4, 0.30141512621741806, 0), (0.0915778218017565, -4, 0.3027230601001204, 0), (0.0915778218017565, -4, 0.3040309939828228, 0), (0.08214772521686953, -4, 0.3053389278655252, 0), (0.08214772521686953, -4, 0.30664686174822753, 0), (0.07108996051964867, -4, 0.3079547956309299, 0), (0.07108996051964867, -4, 0.3092627295136323, 0), (0.05854562734590141, -4, 0.3105706633963347, 0), (0.05854562734590141, -4, 0.31187859727903705, 0), (0.044669219960561946, -4, 0.31318653116173945, 0), (0.044669219960561946, -4, 0.31449446504444184, 0), (0.02962658103961968, -4, 0.3158023989271442, 0), (0.02962658103961968, -4, 0.3171103328098466, 0), (0.01359275595345888, -4, 0.31841826669254897, 0), (0.01359275595345888, -4, 0.31972620057525136, 0), (-0.003250227089996671, -4, 0.3210341344579536, 0), (-0.003250227089996671, -4, 0.3223420683406559, 0), (-0.020715625002271143, -4, 0.32365000222335816, 0), (-0.020715625002271143, -4, 0.32495793610606044, 0), (-0.03861426944291274, -4, 0.3262658699887627, 0), (-0.03861426944291274, -4, 0.32757380387146495, 0), (-0.056756851599403524, -4, 0.32888173775416724, 0), (-0.056756851599403524, -4, 0.3301896716368695, 0), (-0.0749561758036329, -4, 0.3314976055195718, 0), (-0.0749561758036329, -4, 0.3328055394022741, 0), (-0.09302935616990438, -4, 0.3341134732849763, 0), (-0.09302935616990438, -5, 0.3357476434614949, 1), (-0.11079993122729581, -6, 0.33770755214776327, 1), (-0.12809987258581645, -7, 0.33999260233660983, 1), (-0.12809987258581645, -8, 0.3426020979796111, 1), (-0.1447714650087803, -8, 0.3452115936226124, 0), (-0.1447714650087803, -8, 0.3478210892656136, 0), (-0.16066903684779954, -8, 0.3504305849086149, 0), (-0.16066903684779954, -8, 0.3530400805516162, 0), (-0.17566052161369927, -8, 0.35564957619461746, 0), (-0.17566052161369927, -7, 0.35793462638346396, -1), (-0.17566052161369927, -6, 0.3598945350697324, -1), (-0.18962883348575887, -5, 0.3615287052462509, -1), (-0.18962883348575887, -4, 0.3628366391289532, -1), (-0.18962883348575887, -3, 0.36381793830850834, -1), (-0.20247304178003628, -2, 0.3644723038716801, -1), (-0.20247304178003628, -1, 0.3647995364923791, -1), (-0.21410933177999641, 0, 0.3647995364923791, -1), (-0.21410933177999641, 1, 0.3644723038716801, -1), (-0.2244717418524232, 2, 0.36381793830850834, -1), (-0.2244717418524232, 2, 0.3631635727453365, 0), (-0.23351266940018628, 2, 0.3625092071821647, 0), (-0.23351266940018628, 2, 0.3618548416189929, 0), (-0.24120314091137074, 2, 0.36120047605582106, 0), (-0.24120314091137074, 2, 0.3605461104926493, 0), (-0.24753284412106627, 2, 0.35989174492947745, 0), (-0.24753284412106627, 2, 0.3592373793663057, 0), (-0.2525099230768273, 2, 0.35858301380313384, 0), (-0.2525099230768273, 2, 0.35792864823996207, 0), (-0.2561605396603259, 2, 0.35727428267679023, 0), (-0.2561605396603259, 2, 0.3566199171136184, 0), (-0.2585282078350214, 2, 0.3559655515504466, 0), (-0.2585282078350214, 2, 0.3553111859872748, 0), (-0.2596729095323499, 2, 0.354656820424103, 0), (-0.2596729095323499, 2, 0.3540024548609312, 0), (-0.259670003627296, 2, 0.3533480892977594, 0), (-0.259670003627296, 2, 0.35269372373458757, 0), (-0.25860894185987526, 2, 0.3520393581714158, 0), (-0.25860894185987526, 2, 0.35138499260824396, 0), (-0.2565918078049727, 2, 0.35073062704507213, 0), (-0.2565918078049727, 2, 0.35007626148190035, 0), (-0.2537316970540307, 2, 0.3494218959187285, 0), (-0.2537316970540307, 2, 0.34876753035555674, 0), (-0.250150958625065, 2, 0.3481131647923849, 0), (-0.250150958625065, 2, 0.34745879922921313, 0), (-0.2459793192416669, 2, 0.3468044336660413, 0), (-0.2459793192416669, 2, 0.3461500681028695, 0), (-0.2413519134986653, 2, 0.3454957025396977, 0), (-0.2413519134986653, 3, 0.3445144033601426, -1), (-0.2364072440465412, 4, 0.3432064694774403, -1), (-0.23128509676584755, 5, 0.3415722993009218, -1), (-0.23128509676584755, 6, 0.33961239061465337, -1), (-0.2261244364572816, 7, 0.3373273404258068, -1), (-0.2210613088362906, 8, 0.33471784478280553, -1), (-0.2210613088362906, 8, 0.33210834913980425, 0), (-0.2162267745900397, 8, 0.329498853496803, 0), (-0.2162267745900397, 8, 0.32688935785380174, 0), (-0.21174490092936707, 8, 0.32427986221080046, 0), (-0.21174490092936707, 8, 0.3216703665677992, 0), (-0.20773083545239826, 8, 0.31906087092479796, 0), (-0.20773083545239826, 8, 0.3164513752817967, 0), (-0.20773083545239826, 8, 0.3138418796387954, 0), (-0.20428898623643155, 8, 0.3112323839957941, 0), (-0.20428898623643155, 7, 0.3089473338069476, 1), (-0.20151133090033696, 6, 0.30698742512067906, 1), (-0.20151133090033696, 5, 0.30535325494416043, 1), (-0.20151133090033696, 4, 0.30404532106145804, 1), (-0.1994758759439374, 3, 0.30306402188190285, 1), (-0.1994758759439374, 2, 0.3024096563187309, 1), (-0.19824528598936556, 2, 0.301755290755559, 0), (-0.19824528598936556, 2, 0.30110092519238707, 0), (-0.19786570064078074, 2, 0.3004465596292152, 0), (-0.19786570064078074, 2, 0.29979219406604324, 0), (-0.19836575456417663, 2, 0.29913782850287135, 0), (-0.19836575456417663, 2, 0.2984834629396994, 0), (-0.19975581409157045, 2, 0.2978290973765275, 0), (-0.19975581409157045, 2, 0.2971747318133556, 0), (-0.20202744119918098, 2, 0.2965203662501837, 0), (-0.20202744119918098, 2, 0.29586600068701174, 0), (-0.20515309312439373, 2, 0.29521163512383986, 0), (-0.20515309312439373, 2, 0.29455726956066797, 0), (-0.20515309312439373, 1, 0.29423003693996885, 1), (-0.2090860632000084, 0, 0.29423003693996885, 1), (-0.2090860632000084, -1, 0.29455726956066797, 1), (-0.21376066572641134, -2, 0.29521163512383986, 1), (-0.21909266490339166, -3, 0.29619293430339505, 1), (-0.21909266490339166, -3, 0.2971742334829503, 0), (-0.22497994503463187, -3, 0.2981555326625055, 0), (-0.22497994503463187, -3, 0.2991368318420607, 0), (-0.23130341643058586, -3, 0.30011813102161594, 0), (-0.23130341643058586, -3, 0.30109943020117114, 0), (-0.23792814870076068, -3, 0.3020807293807264, 0), (-0.23792814870076068, -3, 0.3030620285602816, 0), (-0.24470472047473604, -3, 0.3040433277398368, 0), (-0.24470472047473604, -3, 0.30502462691939203, 0), (-0.25147077205241, -3, 0.3060059260989472, 0), (-0.25147077205241, -3, 0.3069872252785024, 0), (-0.2580527450863127, -3, 0.30796852445805767, 0), (-0.2580527450863127, -3, 0.30894982363761286, 0), (-0.2642677911694197, -3, 0.30993112281716806, 0), (-0.2642677911694197, -3, 0.3109124219967233, 0), (-0.2699258291658182, -3, 0.3118937211762785, 0), (-0.2699258291658182, -3, 0.3128750203558337, 0), (-0.27483172930188116, -3, 0.31385631953538895, 0), (-0.27483172930188116, -3, 0.31483761871494415, 0), (-0.2787876004530158, -2, 0.3154919842781161, -1), (-0.2787876004530158, -1, 0.3158192168988152, -1), (-0.2787876004530158, 0, 0.3158192168988152, -1), (-0.28159515573364347, 1, 0.3154919842781161, -1), (-0.28305813044122097, 2, 0.31483761871494415, -1), (-0.28305813044122097, 2, 0.31418325315177226, 0), (-0.28298472563116694, 2, 0.3135288875886003, 0), (-0.28298472563116694, 2, 0.3128745220254284, 0), (-0.28119005011783765, 2, 0.3122201564622565, 0), (-0.28119005011783765, 2, 0.3115657908990846, 0), (-0.2774985335132828, 2, 0.31091142533591265, 0), (-0.2774985335132828, 2, 0.31025705977274076, 0), (-0.27174628303323906, 2, 0.3096026942095688, 0), (-0.27174628303323906, 2, 0.30894832864639693, 0), (-0.2637833572182099, 2, 0.308293963083225, 0), (-0.2637833572182099, 2, 0.3076395975200531, 0), (-0.25347593043276406, 2, 0.30698523195688115, 0), (-0.25347593043276406, 2, 0.30633086639370927, 0), (-0.24070832301134915, 2, 0.3056765008305373, 0), (-0.24070832301134915, 2, 0.30502213526736544, 0), (-0.2253848732037147, 2, 0.30436776970419355, 0), (-0.2253848732037147, 2, 0.3037134041410216, 0), (-0.20743162862419134, 2, 0.3030590385778497, 0), (-0.20743162862419134, 2, 0.30240467301467777, 0), (-0.1867978367103466, 2, 0.3017503074515059, 0), (-0.1867978367103466, 2, 0.30109594188833394, 0), (-0.16345721572894134, 2, 0.30044157632516205, 0), (-0.16345721572894134, 2, 0.2997872107619901, 0), (-0.1374089901090858, 2, 0.2991328451988182, 0), (-0.1374089901090858, 2, 0.2984784796356463, 0), (-0.1374089901090858, 1, 0.29815124701494716, 1), (-0.10867867631017297, 0, 0.29815124701494716, 1), (-0.10867867631017297, -1, 0.2984784796356463, 1), (-0.07731860801963147, -2, 0.2991328451988182, 1), (-0.04340819219502817, -3, 0.3001141443783734, 1), (-0.04340819219502817, -3, 0.3010954435579286, 0), (-0.007053890287427089, -3, 0.30207674273748386, 0), (-0.007053890287427089, -3, 0.30305804191703906, 0), (0.03161107812225028, -3, 0.30403934109659425, 0), (0.03161107812225028, -3, 0.3050206402761495, 0), (0.07242730910600975, -3, 0.3060019394557047, 0), (0.07242730910600975, -3, 0.30698323863525995, 0), (0.11521006247706739, -3, 0.30796453781481514, 0), (0.11521006247706739, -3, 0.30894583699437034, 0), (0.15975037814655024, -3, 0.3099271361739256, 0), (0.15975037814655024, -3, 0.3109084353534808, 0), (0.20581645184189723, -3, 0.311889734533036, 0), (0.20581645184189723, -3, 0.31287103371259123, 0), (0.25315525849562537, -3, 0.3138523328921464, 0), (0.25315525849562537, -3, 0.3148336320717016, 0), (0.3014944089436948, -3, 0.31581493125125687, 0), (0.3014944089436948, -3, 0.31679623043081206, 0), (0.35054422304750493, -3, 0.3177775296103673, 0), (0.35054422304750493, -3, 0.3187588287899225, 0), (0.39999999999999974, -3, 0.3197401279694777, 0), (0.39999999999999974, -3, 0.32072142714903284, 0), (0.44954446442015217, -3, 0.321702726328588, 0), (0.44954446442015217, -3, 0.32268402550814307, 0), (0.4988503649047827, -3, 0.3236653246876982, 0), (0.4988503649047827, -3, 0.3246466238672533, 0), (0.5475832000132783, -3, 0.32562792304680843, 0), (0.5475832000132783, -3, 0.3266092222263635, 0), (0.5954040452282119, -3, 0.3275905214059186, 0), (0.5954040452282119, -3, 0.32857182058547374, 0), (0.6419724532783853, -3, 0.3295531197650288, 0), (0.6419724532783853, -3, 0.33053441894458396, 0), (0.686949399343796, -3, 0.33151571812413905, 0), (0.686949399343796, -3, 0.3324970173036942, 0), (0.7300002420936639, -3, 0.33347831648324927, 0), (0.7300002420936639, -3, 0.33445961566280435, 0), (0.7707976712457146, -3, 0.3354409148423595, 0), (0.7707976712457146, -4, 0.3367488487250618, 1), (0.8090246123803834, -5, 0.3383830189015803, 1), (0.8443770600974451, -6, 0.34034292758784873, 1), (0.8443770600974451, -7, 0.34262797777669524, 1), (0.8765668112612082, -8, 0.3452374734196965, 1), (0.8765668112612082, -8, 0.3478469690626978, 0), (0.9053240710370493, -8, 0.3504564647056991, 0), (0.9053240710370493, -8, 0.3530659603487003, 0), (0.9303999056668357, -8, 0.3556754559917016, 0), (0.9303999056668357, -8, 0.35828495163470286, 0), (0.951568517450358, -8, 0.36089444727770414, 0), (0.951568517450358, -8, 0.3635039429207054, 0), (0.9686293191783766, -8, 0.36611343856370665, 0), (0.9686293191783766, -8, 0.36872293420670793, 0), (0.9814087872813331, -8, 0.3713324298497092, 0), (0.9814087872813331, -8, 0.3739419254927105, 0), (0.9897620751946581, -9, 0.37687507171221474, 1), (0.9935743708732034, -10, 0.38013097504346965, 1), (0.9935743708732034, -11, 0.38370864370677976, 1), (0.9927619849876363, -12, 0.38760698790961284, 1), (0.9927619849876363, -13, 0.3918248201785603, 1), (0.9872731590769437, -13, 0.3960426524475078, 0), (0.9872731590769437, -13, 0.40026048471645526, 0), (0.9770885857842073, -13, 0.4044783169854027, 0), (0.9770885857842073, -13, 0.40869614925435016, 0), (0.9622216362371568, -13, 0.4129139815232976, 0), (0.9622216362371568, -13, 0.41713181379224507, 0), (0.9427182926192229, -13, 0.4213496460611925, 0), (0.9427182926192229, -13, 0.42556747833014, 0), (0.9186567869791555, -13, 0.4297853105990875, 0), (0.9186567869791555, -13, 0.43400314286803493, 0), (0.8901469503156669, -13, 0.4382209751369824, 0), (0.8901469503156669, -13, 0.44243880740592983, 0), (0.8573292789160872, -13, 0.4466566396748773, 0), (0.8573292789160872, -13, 0.4508744719438248, 0), (0.820373727793389, -13, 0.45509230421277225, 0), (0.820373727793389, -13, 0.4593101364817197, 0), (0.7794782438235903, -13, 0.46352796875066715, 0), (0.7794782438235903, -13, 0.4677458010196146, 0), (0.7348670538061497, -13, 0.47196363328856206, 0), (0.7348670538061497, -13, 0.47618146555750956, 0), (0.6867887251257352, -12, 0.4800798097603426, -1), (0.6867887251257352, -11, 0.48365747842365275, -1), (0.6867887251257352, -10, 0.4869133817549076, -1), (0.6355140189586049, -9, 0.4898465279744119, -1), (0.6355140189586049, -8, 0.4924560236174132, -1), (0.581333558016828, -7, 0.49474107380625976, -1), (0.581333558016828, -6, 0.49670098249252814, -1), (0.581333558016828, -5, 0.4983351526690467, -1), (0.581333558016828, -4, 0.49964308655174894, -1), (0.524555332637201, -4, 0.5009510204344513, 0), (0.524555332637201, -4, 0.5022589543171535, 0), (0.4655020705797962, -4, 0.5035668881998557, 0), (0.4655020705797962, -4, 0.5048748220825581, 0), (0.40450849718747417, -4, 0.5061827559652603, 0), (0.40450849718747417, -4, 0.5074906898479626, 0), (0.34191851355910635, -4, 0.5087986237306649, 0), (0.34191851355910635, -4, 0.5101065576133671, 0), (0.2780823210954362, -4, 0.5114144914960694, 0), (0.2780823210954362, -4, 0.5127224253787717, 0), (0.2133535211805284, -4, 0.514030359261474, 0), (0.2133535211805284, -4, 0.5153382931441762, 0), (0.14808621885986775, -3, 0.5163195923237314, -1), (0.14808621885986775, -2, 0.5169739578869031, -1), (0.14808621885986775, -1, 0.5173011905076021, -1), (0.08263215916808002, 0, 0.5173011905076021, -1), (0.08263215916808002, 1, 0.5169739578869031, -1), (0.017337924247666836, 2, 0.5163195923237314, -1), (-0.04745778140838772, 3, 0.5153382931441762, -1), (-0.04745778140838772, 4, 0.514030359261474, -1), (-0.11142673135361089, 5, 0.5123961890849554, -1), (-0.1742536383596122, 6, 0.510436280398687, -1), (-0.1742536383596122, 7, 0.5081512302098404, -1), (-0.23563845312819445, 7, 0.5058661800209939, 0), (-0.23563845312819445, 7, 0.5035811298321474, 0), (-0.29529847918259333, 7, 0.5012960796433008, 0), (-0.29529847918259333, 7, 0.4990110294544543, 0), (-0.3529702846691206, 7, 0.49672597926560774, 0), (-0.3529702846691206, 7, 0.49444092907676124, 0), (-0.4084113940906593, 7, 0.4921558788879147, 0), (-0.4084113940906593, 7, 0.4898708286990681, 0), (-0.46140174545523427, 7, 0.4875857785102216, 0), (-0.46140174545523427, 8, 0.48497628286722033, -1), (-0.5117449009293664, 9, 0.48204313664771603, -1), (-0.559269001808168, 10, 0.4787872333164611, -1), (-0.559269001808168, 11, 0.475209564653151, -1), (-0.6038274614223436, 12, 0.4713112204503179, -1), (-0.6452993924655875, 13, 0.4670933881813705, -1), (-0.6452993924655875, 14, 0.46255735263887665, -1), (-0.6835897681127523, 15, 0.4577044955432044, -1), (-0.6835897681127523, 16, 0.4525362951216356, -1), (-0.718629319178377, 16, 0.44736809470006683, 0), (-0.718629319178377, 16, 0.44219989427849804, 0), (-0.750374172404802, 16, 0.43703169385692925, 0), (-0.750374172404802, 16, 0.43186349343536046, 0), (-0.7788052377386661, 16, 0.42669529301379167, 0), (-0.7788052377386661, 16, 0.4215270925922228, 0), (-0.8039273551235625, 16, 0.416358892170654, 0), (-0.8039273551235625, 16, 0.41119069174908524, 0), (-0.8039273551235625, 16, 0.40602249132751644, 0), (-0.8257682138763619, 16, 0.40085429090594765, 0), (-0.8257682138763619, 16, 0.39568609048437886, 0), (-0.8443770600974466, 16, 0.39051789006281007, 0), (-0.8443770600974466, 16, 0.3853496896412413, 0), (-0.8598232097652294, 16, 0.38018148921967243, 0), (-0.8598232097652294, 16, 0.37501328879810364, 0), (-0.8721943871592024, 16, 0.36984508837653485, 0), (-0.8721943871592024, 16, 0.36467688795496606, 0), (-0.8815949100218342, 16, 0.35950868753339726, 0), (-0.8815949100218342, 16, 0.3543404871118285, 0), (-0.8881437443893521, 16, 0.3491722866902597, 0), (-0.8881437443893521, 15, 0.34431942959458745, 1), (-0.8919724532783855, 14, 0.3397833940520936, 1), (-0.8919724532783855, 13, 0.33556556178314617, 1), (-0.8932230643967914, 12, 0.3316672175803131, 1), (-0.8932230643967914, 11, 0.328089548917003, 1), (-0.8932230643967914, 10, 0.32483364558574807, 1), (-0.8920458827423474, 9, 0.32190049936624376, 1), (-0.8920458827423474, 8, 0.3192910037232425, 1), (-0.8920458827423474, 8, 0.3166815080802412, 0), (-0.8885972743556396, 8, 0.31407201243724, 0), (-0.8885972743556396, 8, 0.3114625167942387, 0), (-0.883037447599623, 8, 0.3088530211512374, 0), (-0.883037447599623, 8, 0.30624352550823614, 0), (-0.8755282581475767, 8, 0.3036340298652349, 0), (-0.8755282581475767, 8, 0.30102453422223363, 0), (-0.8662310633764783, 8, 0.29841503857923235, 0), (-0.8662310633764783, 8, 0.29580554293623107, 0), (-0.855304651090192, 8, 0.29319604729322984, 0), (-0.855304651090192, 8, 0.29058655165022856, 0), (-0.8429032664457274, 8, 0.2879770560072273, 0), (-0.8429032664457274, 8, 0.285367560364226, 0), (-0.8291747596384611, 8, 0.2827580647212248, 0), (-0.8291747596384611, 8, 0.2801485690782235, 0), (-0.8142588753340249, 8, 0.2775390734352222, 0), (-0.8142588753340249, 8, 0.27492957779222094, 0), (-0.7982857030335422, 8, 0.2723200821492197, 0), (-0.7982857030335422, 8, 0.26971058650621843, 0), (-0.7813743055457846, 8, 0.26710109086321715, 0), (-0.7813743055457846, 7, 0.2648160406743706, 1), (-0.7813743055457846, 6, 0.2628561319881021, 1), (-0.7636315405374512, 5, 0.2612219618115834, 1), (-0.7636315405374512, 4, 0.2599140279288811, 1), (-0.7451510877663866, 3, 0.2589327287493258, 1), (-0.7451510877663866, 2, 0.25827836318615394, 1), (-0.7260126920987868, 2, 0.257623997622982, 0), (-0.7260126920987868, 2, 0.2569696320598101, 0), (-0.7062816297988399, 2, 0.25631526649663816, 0), (-0.7062816297988399, 2, 0.2556609009334663, 0), (-0.6860084028871816, 2, 0.2550065353702944, 0), (-0.6860084028871816, 2, 0.25435216980712244, 0), (-0.6652286636235525, 2, 0.25369780424395055, 0), (-0.6652286636235525, 2, 0.2530434386807786, 0), (-0.6439633684099625, 2, 0.2523890731176067, 0), (-0.6439633684099625, 2, 0.2517347075544348, 0), (-0.6222191576646015, 2, 0.2510803419912629, 0), (-0.6222191576646015, 2, 0.25042597642809095, 0), (-0.5999889555147108, 2, 0.24977161086491903, 0), (-0.5999889555147108, 2, 0.24911724530174711, 0), (-0.5772527805289216, 2, 0.2484628797385752, 0), (-0.5772527805289216, 2, 0.24780851417540328, 0), (-0.5539787561860056, 2, 0.2471541486122314, 0), (-0.5539787561860056, 2, 0.24649978304905948, 0), (-0.5301243073857074, 2, 0.24584541748588756, 0), (-0.5301243073857074, 2, 0.24519105192271565, 0), (-0.5056375270756029, 2, 0.24453668635954373, 0), (-0.5056375270756029, 2, 0.24388232079637182, 0), (-0.4804586950205747, 2, 0.2432279552331999, 0), (-0.4804586950205747, 2, 0.24257358967002798, 0), (-0.45452192890191956, 2, 0.24191922410685607, 0), (-0.45452192890191956, 2, 0.24126485854368415, 0), (-0.4277569463219396, 2, 0.24061049298051224, 0), (-0.4277569463219396, 1, 0.24028326035981312, 1), (-0.4277569463219396, 0, 0.24028326035981312, 1), (-0.40009091492530763, -1, 0.24061049298051224, 1), (-0.3714503667462531, -2, 0.24126485854368415, 1), (-0.3714503667462531, -3, 0.24224615772323937, 1), (-0.3417631520629205, -4, 0.24355409160594174, 1), (-0.3417631520629205, -4, 0.2448620254886441, 0), (-0.3109604074970865, -4, 0.2461699593713465, 0), (-0.3109604074970865, -4, 0.24747789325404887, 0), (-0.27897851284461656, -4, 0.24878582713675126, 0), (-0.27897851284461656, -4, 0.25009376101945363, 0), (-0.24576101116315402, -4, 0.251401694902156, 0), (-0.24576101116315402, -4, 0.2527096287848584, 0), (-0.2112604669781572, -4, 0.25401756266756076, 0), (-0.2112604669781572, -4, 0.25532549655026315, 0), (-0.17544023809315568, -4, 0.25663343043296555, 0), (-0.17544023809315568, -5, 0.25826760060948417, 1), (-0.13827613739844846, -6, 0.2602275092957527, 1), (-0.09975796225433303, -7, 0.26251255948459923, 1), (-0.09975796225433303, -8, 0.2651220551276005, 1), (-0.059890870467819615, -8, 0.2677315507706018, 0), (-0.059890870467819615, -8, 0.27034104641360307, 0), (-0.018696583569414362, -8, 0.2729505420566043, 0), (-0.018696583569414362, -8, 0.2755600376996056, 0), (0.023785599989075652, -8, 0.27816953334260686, 0), (0.023785599989075652, -8, 0.28077902898560814, 0), (0.06749799697522088, -8, 0.28338852462860936, 0), (0.06749799697522088, -8, 0.28599802027161064, 0), (0.11236394981292483, -8, 0.2886075159146119, 0), (0.11236394981292483, -8, 0.2912170115576132, 0), (0.15828746581516323, -8, 0.29382650720061443, 0)]

n = len(data)

X = torch.zeros([n, 3], dtype=torch.float)
y = torch.zeros([n, 3], dtype=torch.float)
for i in range(n):
	X[i][0] = data[i][0]
	X[i][1] = data[i][1] / 90
	X[i][2] = data[i][2]
	y[i][data[i][3] + 1] = 1


model = torch.nn.Sequential(
	torch.nn.Linear(3, 5),
	torch.nn.ReLU(),
	torch.nn.Linear(5, 3))

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(10000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([0.10610737385376351, 0, 0.5])))
print(model.eval())

torch.save(model.state_dict(), "nn_state.pt")


