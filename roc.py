#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

def smart_sort(x,y):
    xy=[(x[i],y[i]) for i in range(len(x))]
    xy.sort()
    out=([],[])
    previous=0
    for i in range(len(xy)):
        if xy[i][1]>=previous:
            previous=xy[i][1]
            out[0].append(xy[i][0])
            out[1].append(xy[i][1])
    return out

# raspi 3

#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
fpr1,tpr1=[[0.12269546403517669, 0.11127432657424419, 0.10419912905997404, 0.09549127006827142, 0.09222920949164898, 0.08917266697890142, 0.08601717742486764, 0.0826888053547873, 0.0812730071515012, 0.07896768059359915, 0.07798504083770626, 0.07649822161812068, 0.07525009049570001, 0.07457748760379046, 0.07457118470919519, 0.07064688992246537, 0.07020152570959193, 0.06951565648513826, 0.06951565648513826, 0.06899153466294615], [0.007639759104622975, 0.005367709405203584, 0.0045708187007464166, 0.0038759027965001272, 0.0032217851521296596, 0.0027410948613625663, 0.0025128917573780806, 0.0023504409593157046, 0.0022476849986370487, 0.0021452014711464433, 0.001959782887373108, 0.001914179517644781, 0.0019008052720558889, 0.0017828458715257152, 0.0017449568089080048, 0.0017128736311193115, 0.0016860195025828952, 0.0016688047830216368, 0.00164897831876473, 0.0016093712591040065], [0.005930801112623147, 0.004546398213200567, 0.003789185519958373, 0.003185974344285723, 0.0028271771524039807, 0.002688697491429952, 0.002508481773899881, 0.0023492363111864635, 0.002181570973433026, 0.0019453848445622755, 0.0019090350214403657, 0.0017490270040063424, 0.0015845900908580772, 0.001549636903284233, 0.001397167402354457, 0.0013506985333476998, 0.001305312713074556, 0.0012575093733067392, 0.0012459923104341148, 0.0012181203828200969]],[[0.9946144913037752, 0.9943286866334229, 0.9936542407173691, 0.9930134473392659, 0.9924733190420872, 0.9919480086628226, 0.9911983480305533, 0.9907884193887292, 0.9904023265548785, 0.9900216230776572, 0.9890677463336947, 0.9887523789607461, 0.9862228129812184, 0.9835346772779836, 0.9824720907975852, 0.9797101017813753, 0.9783910314309923, 0.977668874519559, 0.9769971232818198, 0.9733946702874592], [0.6810121859680652, 0.6743060925373363, 0.6701713555041454, 0.6668116364978705, 0.6648942584533186, 0.6631036445516916, 0.6620737468888684, 0.6581986004480364, 0.65752498998171, 0.6560118345232064, 0.6560118345232064, 0.6539003875953497, 0.6539003875953497, 0.6519442520803893, 0.6494863715502074, 0.6494863715502074, 0.6488365035536471, 0.6463617311477423, 0.644129619920972, 0.6421923282318179], [0.005880177411054864, 0.004384784994880121, 0.003436976268418219, 0.002991706207836962, 0.002991706207836962, 0.0026921772743636915, 0.0026921772743636915, 0.0024648637040977643, 0.0022779678052158164, 0.0022779678052158164, 0.002051596085448199, 0.002051596085448199, 0.002051596085448199, 0.002051596085448199, 0.002051596085448199, 0.002051596085448199, 0.0015891474907782086, 0.0015891474907782086, 0.0015891474907782086, 0.0015891474907782086]]

#[0,0.00001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
fpr2,tpr2=([[0.17228012908119716, 0.17227932231068896, 0.1710708691703244, 0.15366216840956856, 0.1450192746130383, 0.13983898605297668, 0.13621002473771837, 0.13312193749810042, 0.1305880242821048, 0.12815385479327226, 0.12602198221386007, 0.12432498414994289], [0.018634977708967517, 0.018634977708967517, 0.018533433797982637, 0.015391674504877596, 0.01310560005613458, 0.011741112873922177, 0.01065815342235943, 0.009845001514499182, 0.009236784692419505, 0.008765828328264239, 0.008278053384642584, 0.007912498085027506], [0.009284123063528722, 0.009284123063528722, 0.009272850515484987, 0.008954590290489929, 0.008525987568559419, 0.008090597833327577, 0.0076595333858316345, 0.007296441297887641, 0.006931920801282222, 0.006646504305421942, 0.006441943475860654, 0.006167378188932179]], [[0.9954954148360786, 0.9954954148360786, 0.9954900142281827, 0.995322871039441, 0.9951752713005221, 0.9950823639678147, 0.9949734967136485, 0.9949064166630761, 0.9948173122584286, 0.994742243808677, 0.994742243808677, 0.9946144913037752], [0.6903329165356843, 0.6903329165356843, 0.6902968805341422, 0.6887393745174906, 0.6873125365439309, 0.686398948829835, 0.685821434367621, 0.6850897533863096, 0.6842399794249446, 0.6836617142126985, 0.6824338249976525, 0.6819377856826752], [0.011492295980286496, 0.011492295980286496, 0.011482467975086744, 0.01099010538959001, 0.010303250676192352, 0.009603435280937522, 0.008919569919121455, 0.00831481999916339, 0.007673501709857917, 0.0071104593869663005, 0.0067468231945754815, 0.006294059280029414]])
#fpr2,tpr2=([[0.134180936401822], [0.0022563764513150097], [0.003997944544802125]], [[0.9586638990576428], [0.7496904798383169], [0.00362309411688852]])


#[5,10,30,100,1000,10000,100000,1000000]
fpr3,tpr3=([[0.06555338969982276, 0.06521212913640043, 0.06294938829592779, 0.05505829200131766, 3.524578657673143e-06], [0.0010494835285984078, 0.0007022674304279202, 1.267648303582376e-06, 0, 0], [0.0008703995745430926, 0.0004308144069082087, 0.0004308144069082087, 0.0004308144069082087, 0]], [[0.908591161589018, 0.8059642704670115, 0.7618282356123431, 0.700065336666836, 0.682226571849319], [0.5694826380702976, 0.5423135574701308, 0.002239205814574014, 0, 0], [0.0007949627705949279, 0.0007949627705949279, 0.0007949627705949279, 0.0007949627705949279, 0]])

# raspi 1
#[0,0.00001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0, 5,10,30,100,1000,10000,100000,1000000]:
fprA,tprA=([[0.010623533882637712, 0.010623533882637712, 0.010596481859034827, 0.009586842183467705, 0.008463456270104669, 0.007551492972273367, 0.00679699446992264, 0.00619286958643984, 0.005836433333603994, 0.005506850773287767, 0.005243892330001348, 0.005087708282702531, 0.00489256730448929, 0.003505259865545678, 0.002826760825838731, 0.002291071931485391, 0.0017172748960231229, 0.0013671650685890319, 0.001170005483333413, 0.0010030325612593914, 0.0008241177548922047, 0.0007744173300436616, 0.0006380109255988456, 0.0006065754489014394, 0.0005764156780701151, 0.0005494426507461577, 0.0005033172277120499, 0.0004796267478933075, 0.00046180888506546153, 0.00044924175362844574, 0.000434005556426821, 0.0004264118290184437, 0.00011877510729698999, 0.0001069760886146507, 3.0415248158919025e-05, 0, 0, 0, 0, 0], [0.006543952204299249, 0.006543952204299249, 0.006531275721263425, 0.006332977719528473, 0.006072552441230779, 0.00579089961370906, 0.0055024220568336245, 0.005248311390644674, 0.004991793304651113, 0.0047390976230817315, 0.004512183016879506, 0.0043357046895630144, 0.004118291885776676, 0.0031253660430036706, 0.0026263198712090452, 0.0022347374723784195, 0.001979202091813184, 0.0017575193147937257, 0.0015754727867051393, 0.0014137308709177932, 0.0012267166070430186, 0.0011160030953358438, 0.0010392214151934201, 0.0010392214151934201, 0.0009944826038475803, 0.0009637240629375207, 0.0009450359802163968, 0.0009156201457331579, 0.0008745772519741881, 0.0008476230459401208, 0.0008090793096919418, 0.0007543897371560293, 0.0003670453423578847, 0.00016011982634624885, 0, 0, 0, 0, 0, 0], [0.010117459992058904, 0.010117459992058904, 0.010107513626137963, 0.009801781677884158, 0.009300788752205146, 0.008722788851262661, 0.008135730850133563, 0.007616530549060383, 0.007036834240152494, 0.006499899016066389, 0.0061337525649208385, 0.005849665014062773, 0.005530111761497992, 0.0037438922561458827, 0.0029651829795570533, 0.00243024117297427, 0.002054343138907063, 0.0018401702518349536, 0.0017170908807618505, 0.0016255594483743803, 0.0015573867800643005, 0.0015225952209366714, 0.001467236787388072, 0.001467236787388072, 0.0014559131260750216, 0.001410029987505831, 0.0013763808790193991, 0.0013274710060428773, 0.0012911488117151922, 0.0012694505381707147, 0.0012694505381707147, 0.0012694505381707147, 0.0010431610433913018, 0.0006169274905655714, 0.00043230774323606123, 0, 0, 0, 0, 0]], [[0.24440147810812676, 0.24440147810812676, 0.2443518093923851, 0.2430873751918812, 0.24225752115799248, 0.24168386408679401, 0.24110276992847238, 0.2405255743339755, 0.24031878730789863, 0.2399419205125366, 0.23976156271134977, 0.2395114189300094, 0.23933586541959653, 0.23704717217785481, 0.2352615511872401, 0.23342050083246152, 0.23207059076074016, 0.23029464023363885, 0.22935187411529534, 0.22743431514865958, 0.22651600365794425, 0.22572295251661212, 0.22238722767033858, 0.2190578879177752, 0.2167181645570514, 0.2125712065228632, 0.2087319818773376, 0.20222386119420593, 0.189274699878515, 0.17905556835668396, 0.1743320954296191, 0.16952180210499607, 0.12275893969797674, 0.11994093062611436, 0.059245281810948944, 0, 0, 0, 0, 0], [0.5815006445845932, 0.5815006445845932, 0.5815006445845932, 0.5814226041187536, 0.5811097478116153, 0.5808757953328536, 0.5806835657871273, 0.5804399098642004, 0.5801929131036305, 0.5800293059028792, 0.5798239194628398, 0.5797275419274655, 0.5795287057814565, 0.5786303207555112, 0.5779586997767701, 0.5772723640973992, 0.5772033514006959, 0.5771161142469627, 0.5771161142469627, 0.5764910960764659, 0.5764910960764659, 0.5764776951883923, 0.5764776951883923, 0.5764776951883923, 0.5764776951883923, 0.5764776951883923, 0.5760804170401413, 0.5760804170401413, 0.5760804170401413, 0.5760804170401413, 0.5760804170401413, 0.5760804170401413, 0.5521870652864053, 0.15228116054168192, 0, 0, 0, 0, 0, 0], [0.011995387471459626, 0.011995387471459626, 0.011985559466259874, 0.011473397545287806, 0.010627677222838326, 0.009771002769593291, 0.008664103208960406, 0.007505217595822999, 0.006930873066951668, 0.006215353338388067, 0.005938592616960889, 0.0057154354738940235, 0.005600345438002763, 0.005201389851925336, 0.004746209886100996, 0.004687241854902485, 0.004404952880550445, 0.004404952880550445, 0.004404952880550445, 0.004404952880550445, 0.004404952880550445, 0.004404952880550445, 0.00395220944101521, 0.00395220944101521, 0.00395220944101521, 0.00395220944101521, 0.00395220944101521, 0.00395220944101521, 0.0031580656708535947, 0.0031580656708535947, 0.0031580656708535947, 0.0031580656708535947, 0.0025935286721711823, 0.0012917274834207183, 0.0007993444229131506, 0, 0, 0, 0, 0]])


fprA2,tprA2=([[0.09455804332523353, 0.09455804332523353, 0.09422798006415617, 0.08769790729764146, 0.06851930331137795, 0.04650481526944556, 0.02709660103500622, 1.2249465549414338e-05], [0.019568647943374273, 0.019568647943374273, 0.019527282577678426, 0.018880274605536932, 0.011483416097400724, 0.0030894396113121205, 0.00020821540375914072, 0], [0.04275179602116546, 0.04275179602116546, 0.042674434015896205, 0.040230439229156804, 0.019326301498523497, 0.007364449574872083, 0.0021868895163213073, 0.00043230774323606123]], [[0.5040724493349424, 0.5040724493349424, 0.5032154628720106, 0.49608082104226703, 0.4765159769361864, 0.4220184257827676, 0.1981175989907659, 0.04043619089792009], [0.6834635349729676, 0.6834635349729676, 0.6833824539694979, 0.6816841823218225, 0.6710054575585898, 0.6547786149829323, 0.2195390541198994, 0], [0.04670665286132236, 0.04670665286132236, 0.04665077655675961, 0.04449770631762397, 0.018239897225273624, 0.006308248462536556, 0.002867177191951773, 0.0008091724281129024]])

#[0,0.00001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0, 5,10,30,100,1000,10000,100000,1000000]:
fprC,tprC=([[0.018144041825080746, 0.018144041825080746, 0.018035675738111354, 0.015056243678228236, 0.013147271705598726, 0.011805575373490246, 0.010967599654347916, 0.010353339716355927, 0.009754413460335303, 0.009143262950343647, 0.008729169498522192, 0.00845996026456916, 0.008063785521889076, 0.005819417198968666, 0.004994380061851498, 0.0045637898339093656, 0.004094121659476579, 0.003808059325761751, 0.0035828913576966587, 0.0031384314001011555, 0.003020026062620421, 0.003020026062620421, 0.003013148344038066, 0.002871993758998149, 0.0028542364039584175, 0.002847994017151265, 0.002713775977709926, 0.002713775977709926, 0.002713775977709926, 0.0025464954743796555, 0.0025464954743796555, 0.0025464954743796555, 0.0019561344319210263, 0.0015303612962215695, 0.0, 0, 0, 0, 0, 0], [0.026434901118824586, 0.026434901118824586, 0.026192920679330114, 0.019969563958826122, 0.015737540946795036, 0.01310261858068372, 0.011469585943211421, 0.010267053341468874, 0.009311917873781174, 0.008498307277390022, 0.007736232422393525, 0.00710180502568594, 0.006549267431396726, 0.004051877956397871, 0.003153309904292289, 0.0026573272158979875, 0.002261077313774128, 0.0019978957343952885, 0.001819685510383333, 0.0016792823410396237, 0.0016178125180378396, 0.0014156420731298827, 0.001271505457137684, 0.001100406295319947, 0.0010296237051756172, 0.0009732550546235561, 0.0008947803968125322, 0.0008947803968125322, 0.0008629335131155153, 0.0005866918202931892, 0.0005354521414957543, 0.0005337925229930379, 0.00025375900498192594, 0.00013360762926014114, 0, 0, 0, 0, 0, 0], [0.007083550386867563, 0.007083550386867563, 0.0070572477747655165, 0.006426481021172723, 0.0056233644677656704, 0.005023727076626003, 0.004648761514363885, 0.004319701193706337, 0.004143831485507833, 0.003999821921369703, 0.003873863629938372, 0.0037716840602566486, 0.0036882257615416763, 0.0028701882778134993, 0.0022996578213897007, 0.0020316571359690477, 0.001836770528705587, 0.0017189420099749148, 0.0015340556447811348, 0.0014884280725585238, 0.0013171958574706595, 0.0011675265348355711, 0.0010923389156718046, 0.0009683574644672604, 0.0009276105187444675, 0.0008638626018294956, 0.0008191412540574594, 0.0008191412540574594, 0.0007529288486976319, 0.0007529288486976319, 0.0007529288486976319, 0.0007529288486976319, 0.0004304925314554893, 0.0004304925314554893, 0.0004304925314554893, 0, 0, 0, 0, 0]], [[0.08777176526975612, 0.08777176526975612, 0.08769230882608915, 0.08559526403141642, 0.0841191316270304, 0.08307389459949731, 0.08202410080905213, 0.08105540614717556, 0.08019083070502343, 0.07966922761805395, 0.07898341229725281, 0.07855116676842594, 0.07831155417248238, 0.0757050970380383, 0.07431342226525607, 0.07223713043154958, 0.07112423953885515, 0.06919261648915741, 0.06853939046162748, 0.06853939046162748, 0.06784934466464129, 0.06784934466464129, 0.06757933677238419, 0.06718074940714124, 0.06718074940714124, 0.06694458532436448, 0.06694458532436448, 0.06694458532436448, 0.06694458532436448, 0.06694458532436448, 0.06694458532436448, 0.06694458532436448, 0.051484388864988385, 0.043439992132664446, 0.0013453251806431615, 0, 0, 0, 0, 0], [0.539562227521141, 0.539562227521141, 0.5395021675185708, 0.537650029651811, 0.5360578765211768, 0.5348127575928935, 0.5335917377406414, 0.5329302331185831, 0.5324718251489662, 0.5321862773742465, 0.531977287334053, 0.5314576181618145, 0.5312923405422416, 0.5281757707026219, 0.5268950662228158, 0.5254483709109062, 0.5231502124500594, 0.5226520147287397, 0.5226520147287397, 0.5226520147287397, 0.5226520147287397, 0.5226520147287397, 0.5223570075036152, 0.5220111932700665, 0.5217861747166871, 0.5217861747166871, 0.5217861747166871, 0.5217861747166871, 0.5217861747166871, 0.5217861747166871, 0.5195547767024472, 0.5138225937508957, 0.3528287726301163, 0.23935782268676883, 0, 0, 0, 0, 0, 0], [0.008084230427164213, 0.008084230427164213, 0.008054746411564957, 0.007245348758332894, 0.006264513839397659, 0.005306754257671009, 0.004727106700993978, 0.0038857270808410555, 0.003295985343823446, 0.003064310596250129, 0.0028972345078543474, 0.002789126450657077, 0.002789126450657077, 0.0018885535741864823, 0.0017183448091332798, 0.0016888607935340244, 0.0015733203074044415, 0.0015733203074044415, 0.0015733203074044415, 0.0015733203074044415, 0.0015733203074044415, 0.0015733203074044415, 0.0015733203074044415, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0.0007892707175834049, 0, 0, 0, 0, 0]])

fprC2,tprC2=([[0.19051930427614103, 0.19051930427614103, 0.19002665658839984, 0.1738619690589291, 0.1308345615840057, 0.09141403008167881, 0.04953640504268343, 0], [0.08254078013125411, 0.08254078013125411, 0.08196935151931799, 0.06477274042527341, 0.02368197487329525, 0.004605250919799424, 0.00025809152665107757, 0], [0.025783976809815774, 0.025783976809815774, 0.025742865164009213, 0.024128032594992625, 0.010756668723727098, 0.00578407978233185, 0.002879829345558257, 0]], [[0.16322459203709508, 0.16322459203709508, 0.16301870511233413, 0.15653959019410948, 0.14072261044998388, 0.12122569586509263, 0.06588112687081628, 0], [0.6059499830433721, 0.6059499830433721, 0.6057124645019578, 0.5999019595533045, 0.5740111998015918, 0.5495197067285089, 0.2880911844472521, 0], [0.03675581807158445, 0.03675581807158445, 0.03673616206118494, 0.035018554352441646, 0.014160660817051621, 0.004664862668062216, 0.0007990987227831567, 0]])

fprC3,tprC3=([[0.3212120725817878, 0.3212120725817878, 0.3202886985235812, 0.2972210522602522, 0.22940037691276063, 0.1745630912883033, 0.12939536031289436, 0.04503784764667724], [0.14885363818725686, 0.14885363818725686, 0.14828011628766152, 0.127494077014074, 0.05461103671963065, 0.009858582864909602, 0.002024448240473506, 0], [0.062236414654916423, 0.062236414654916423, 0.06211398041618847, 0.05616582703935241, 0.018388352284980095, 0.00799373128077529, 0.003766028446078824, 0]], [[0.22892954095221055, 0.22892954095221055, 0.2283089604742841, 0.21420368090019942, 0.18199829771714, 0.14430056943728373, 0.09258774048393621, 0.03238152175932768], [0.7785689619553855, 0.7785689619553855, 0.7784752495826253, 0.7752443592818633, 0.7433308012411625, 0.6856298241906713, 0.3388690825389788, 0], [0.07956994614846925, 0.07956994614846925, 0.07948227215208314, 0.07344963781036712, 0.028301931798844566, 0.007297109585718254, 0.0014141475731901272, 0]])

fpr4,tpr4=([[0.34604021815188835, 0.34604021815188835, 0.34182692472294823, 0.2740196636159273, 0.19672487713654113], [0.06102817544478808, 0.06102817544478808, 0.06076960299016218, 0.05237208348266367, 0.026049313640096317], [0.03257277769413391, 0.03257277769413391, 0.03250147468343815, 0.03039403176491037, 0.015580512525654079, 0.007873866519910247, 0.002515085055719618]], [[0.9938660908133482, 0.9938660908133482, 0.9938660908133482, 0.9937577017380061, 0.9935890058744944], [0.7897342102769384, 0.7897342102769384, 0.7897252012765529, 0.7886626647560829, 0.7826739505185534], [0.03930995282291329, 0.03930995282291329, 0.03925098479171478, 0.036818471604732866, 0.01952302520414456, 0.008458124599982273, 0.0007949627705949279]])

fprA3,tprA3=([[0.25365182107356943, 0.25365047645605576, 0.2497991338925294, 0.20757613712206366, 0.14200464382819142, 0.07765864838244119, 0.061291518601658025, 0.011715074210961678], [0.03861246307972578, 0.03861246307972578, 0.038524792411852804, 0.03732496190254684, 0.024395983340149006, 0.006446296026105071, 0.0011853053724940827, 0], [0.07327895298672908, 0.07327895298672908, 0.07319604173813976, 0.06939745275806697, 0.031035510202024132, 0.011376676824848738, 0.004248246224642346, 0.00043230774323606123]], [[0.6699338375464629, 0.6699338375464629, 0.6692360396268929, 0.6559811757761336, 0.6097794196525882, 0.5630897411081902, 0.3868094607646596, 0.2600918529765284], [0.7954665058409948, 0.7954665058409948, 0.7954394788398382, 0.7944818220988564, 0.7865471074718002, 0.7615850630473308, 0.2353195758014571, 0], [0.0853014954558887, 0.0853014954558887, 0.08520321540389118, 0.08151451935229433, 0.03383808807788645, 0.008456548024148147, 0.003226861707251858, 0.0008091724281129024]])

fprA4,tprA4=([[0.4704710472309625, 0.4704710472309625, 0.4666300447761162, 0.4156585176957035, 0.31171306585662784, 0.18000911996913269, 0.10128873806710154, 0.02572759384042348], [0.07235049040037947, 0.07235049040037947, 0.07197976582001767, 0.06895222831089858, 0.05001752405632657, 0.013920898070333351, 0.002275510712879829, 0], [0.1375576963993839, 0.1375576963993839, 0.1374584441004503, 0.13142472572933964, 0.057319088827729694, 0.017471848541514012, 0.006195672837097837, 0.00043230774323606123]], [[0.8421606803825343, 0.8421606803825343, 0.8413188887549425, 0.8238491148462728, 0.7753138741892499, 0.7067023574840474, 0.47937343294822105, 0.3660224140866936], [0.83151161022725, 0.83151161022725, 0.8315026012268645, 0.8308357662795781, 0.8276000899473613, 0.8173467401523321, 0.33383584786733783, 0], [0.15947533439941255, 0.15947533439941255, 0.15935674313666887, 0.1536010947414992, 0.07097844732792202, 0.01663244507481088, 0.005267626511969496, 0.0008091724281129024]])



fprC4,tprC4=([[0.5960258841089998, 0.5960258841089998, 0.594713049991822, 0.5543272881688602, 0.432296439997562, 0.29794946854438176, 0.22842360809308676, 0.15764743991977084], [0.27156838642419334, 0.27156771924087564, 0.2708578209012518, 0.24643570760445913, 0.14581156720153443, 0.03521591064749642, 0.0049965706151987145, 0], [0.14981984845034452, 0.14981984845034452, 0.149635745361467, 0.13833327118926791, 0.04624131687608181, 0.013195437524038863, 0.004484772187682062, 0]], [[0.4264482350298651, 0.4264482350298651, 0.42436652571136313, 0.3801955950538015, 0.272869027433271, 0.18630318977848095, 0.1290958048544793, 0.06025178635372934], [0.8930959919689266, 0.8930959919689266, 0.8930388786602325, 0.8907537457124431, 0.8748704838889888, 0.814825740313199, 0.39281463901001146, 0], [0.17976250748284864, 0.17976250748284864, 0.17958296211285568, 0.16951924858839645, 0.07036485220328417, 0.01450109882216886, 0.0026809364934164753, 0]])
band=int(sys.argv[1])

# fprB,tprB=[1]+fprB[band]+[0],[1]+tprB[band]+[0]
fprB,tprB=[1]+fpr4[band]+fpr2[band]+fpr1[band]+fpr3[band]+[0],[1]+tpr4[band]+tpr2[band]+tpr1[band]+tpr3[band]+[0]
fprA,tprA=[1]+fprA2[band]+fprA[band]+[0]+fprA3[band]+fprA4[band],[1]+tprA2[band]+tprA[band]+[0]+tprA3[band]+tprA4[band]
fprC,tprC=[1]+fprC[band]+[0]+fprC2[band]+fprC3[band]+fprC4[band],[1]+tprC[band]+[0]+tprC2[band]+tprC3[band]+tprC4[band]

fprA,tprA = smart_sort(fprA,tprA)
fprB,tprB = smart_sort(fprB,tprB)
fprC,tprC = smart_sort(fprC,tprC)

plt.title('ROC curves')
plt.plot(fprA, tprA, 'ro-', label = 'Probe 1 (AUC = %0.3f)' % metrics.auc(fprA,tprA))
plt.plot(fprB, tprB, 'bo-', label = 'Probe 2 (AUC = %0.3f)' % metrics.auc(fprB,tprB))
plt.plot(fprC, tprC, 'go-', label = 'Probe 3 (AUC = %0.3f)' % metrics.auc(fprC,tprC))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()