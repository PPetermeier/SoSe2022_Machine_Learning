"?W
BHostIDLE"IDLE1ffff???@Affff???@a?<???i?<????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1%???J?@9%???J?@A%???J?@I%???J?@a?H????i1"E??Y???Unknown?
?Host
_MklMatMul")gradient_tape/sequential/dense_1/MatMul_1(1'1??o@9'1??o@A'1??o@I'1??o@a??\?M??i?&????Unknown
?HostRandomStandardNormal"<sequential/gaussian_noise/random_normal/RandomStandardNormal(1u?V?k@9u?V?k@Au?V?k@Iu?V?k@a???@?Y??i
?*@?????Unknown
?HostRandomStandardNormal">sequential/gaussian_noise_1/random_normal/RandomStandardNormal(1d;?O??j@9d;?O??j@Ad;?O??j@Id;?O??j@a??~K???irJ!??j???Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_1/MatMul(1?~j?t?f@9?~j?t?f@A?~j?t?f@I?~j?t?f@a%??u???i?W??????Unknown
}Host
_MklMatMul"%gradient_tape/sequential/dense/MatMul(1??Q??f@9??Q??f@A??Q??f@I??Q??f@a?\?CT??i?7!??????Unknown
zHost_MklNativeFusedMatMul"sequential/dense_1/Relu(17?A`??d@97?A`??d@A7?A`??d@I7?A`??d@a Lb#?	??i*J< ???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1P??nsd@9P??nsd@AP??nsd@IP??nsd@a??m?OL??iq "@8???Unknown
x
Host_MklNativeFusedMatMul"sequential/dense/Relu(1??C?lo]@9??C?lo]@A??C?lo]@I??C?lo]@a|??????iϛ?HR????Unknown
}Host_MklNativeFusedMatMul"sequential/dense_2/BiasAdd(1u?V~P@9u?V~P@Au?V~P@Iu?V~P@a?a0P>y?i^??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1+??N@9+??N@A+??N@I+??N@a?u??³w?i??wn6;???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1V-bG@9V-bG@AV-bG@IV-bG@a??
???q?i??EP _???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1????MRF@9????MRF@A????MRF@I????MRF@a;<
??q?i??*????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1#??~j|@@9#??~j|@@A#??~j|@@I#??~j|@@aɷ?_;i?i???we????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1J+??@9J+??@AJ+??@IJ+??@a??8J?g?i?? ?/????Unknown
iHostWriteSummary"WriteSummary(1????K?;@9????K?;@A????K?;@I????K?;@a????5e?iC??L????Unknown?
?Host
_MklMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?"??~J;@9?"??~J;@A?"??~J;@I?"??~J;@a?pp?d?iIS?h/????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??ʡE9@9??ʡE9@A??ʡE9@I??ʡE9@a1&,9?2c?io?b????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1D?l??i<@9D?l??i<@AZd;߯8@IZd;߯8@a???M?b?i??J`F???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#??~j<8@9#??~j<8@A#??~j<8@I#??~j<8@a??8-??b?i??wS????Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1?|?5^Z7@9?|?5^Z7@A?|?5^Z7@I?|?5^Z7@a2G??a?i?۾J?&???Unknown
^HostGatherV2"GatherV2(1Zd;??3@9Zd;??3@AZd;??3@IZd;??3@a?{Ŋ??^?i?>?5???Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_2/MatMul(1Zd;??3@9Zd;??3@AZd;??3@IZd;??3@a?{Ŋ??^?in?I?4E???Unknown
oHostSoftmax"sequential/dense_2/Softmax(1d;?O??3@9d;?O??3@Ad;?O??3@Id;?O??3@a?y??O-^?i+?2uKT???Unknown
cHostDataset"Iterator::Root(1?z?G?F@9?z?G?F@A??Mb?,@I??Mb?,@a??r??U?i?N6O<_???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?ʡE??0@9?ʡE??0@A?O??n?&@I?O??n?&@a?Ć??vQ?i????g???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?C?l?;"@9?C?l?;"@A?C?l?;"@I?C?l?;"@a??TB?K?ig??n???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(11?Zd!@91?Zd!@A1?Zd!@I1?Zd!@aA?1?Z?J?i?s?n?u???Unknown
eHost
LogicalAnd"
LogicalAnd(1'1?Z!@9'1?Z!@A'1?Z!@I'1?Z!@a?䔎??J?i?j=|???Unknown?
`HostGatherV2"
GatherV2_1(1???x?? @9???x?? @A???x?? @I???x?? @aQ$^Bj|I?iG??4?????Unknown
[ HostAddV2"Adam/add(133333s @933333s @A33333s @I33333s @a???D-I?i?q:??????Unknown
?!HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??ʡE @9??ʡE @A??ʡE @I??ʡE @a58??H?i?5?g!????Unknown
g"HostStridedSlice"strided_slice(1     @ @9     @ @A     @ @I     @ @a??V??H?i ?"Y????Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1+???@9+???@A+???@I+???@aGƧ2уD?i??]z????Unknown
Z$HostArgMax"ArgMax(1?n??J@9?n??J@A?n??J@I?n??J@a??t??D?i*zŁ????Unknown
V%HostMean"Mean(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a?x?p?:C?iH??qP????Unknown
l&HostIteratorGetNext"IteratorGetNext(15^?I@95^?I@A5^?I@I5^?I@a0aa>0#C?i?E?=????Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1R???Q@9R???Q@AR???Q@IR???Q@a?Kϡ?A?is??c^????Unknown
?(HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1D?l??)@9D?l??)@AD?l??)@ID?l??)@a?????@?i?0-曱???Unknown
w)HostDataset""Iterator::Root::ParallelMapV2::Zip(1?z?G?J@9?z?G?J@A??(\??@I??(\??@a,2????@?ik??ŵ???Unknown
?*HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1?t??@9?t??@A?t??@I?t??@aچˠ ?<?i??.#`????Unknown
V+HostSum"Sum_2(1?ʡE??@9?ʡE??@A?ʡE??@I?ʡE??@a??Æ?y;?iVĿ[ϼ???Unknown
?,HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1d;?O??@9d;?O??@Ad;?O??@Id;?O??@a?CAX??:?i~̪?,????Unknown
|-HostMul"+sequential/gaussian_noise/random_normal/mul(1?x?&1@9?x?&1@A?x?&1@I?x?&1@ak??M:?i`[o????Unknown
?.HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(17?A`??@97?A`????A7?A`??@I7?A`????a?ŬU?6?i????I????Unknown
p/HostAddV2"sequential/gaussian_noise/add(1^?I+@9^?I+@A^?I+@I^?I+@a!E7'R6?i??g?????Unknown
[0HostPow"
Adam/Pow_1(1V-???@9V-???@AV-???@IV-???@a?????$6?i?|??????Unknown
r1HostAddV2"sequential/gaussian_noise_1/add(1V-?
@9V-?
@AV-?
@IV-?
@a?Z?6?m4?i?V??e????Unknown
~2HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a?B?v?2?i?ɥ?????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a???g2?i???????Unknown
Y4HostPow"Adam/Pow(1)\???(@9)\???(@A)\???(@I)\???(@a???(?1?izמ?:????Unknown
~5HostMul"-sequential/gaussian_noise_1/random_normal/mul(1ˡE???@9ˡE???@AˡE???@IˡE???@aЙ?@o0?i-솼<????Unknown
\6HostArgMax"ArgMax_1(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@a;??ݫ.?i?gcz'????Unknown
]7HostCast"Adam/Cast_1(1???Q?@9???Q?@A???Q?@I???Q?@a?:Ӻ|..?i?/b
????Unknown
?8HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1-???g3@9-???g3@Ao??ʡ@Io??ʡ@aZmz
.?i??O"?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_6(1R???Q@9R???Q@AR???Q@IR???Q@a???jÑ-?i?g?>?????Unknown
?:HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1?G?z@9?G?z@A?G?z@I?G?z@aU???3-?iQx0z?????Unknown
T;HostSub"sub(1??Q?@9??Q?@A??Q?@I??Q?@apVz??+?i??3S????Unknown
X<HostEqual"Equal(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a?G???+?i??t?????Unknown
?=HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1Zd;?O?@9Zd;?O?@AZd;?O?@IZd;?O?@ah??6?*?i?'(??????Unknown
V>HostCast"Cast(1??n?? @9??n?? @A??n?? @I??n?? @a??e???)?i.???T????Unknown
X?HostCast"Cast_2(1y?&1? @9y?&1? @Ay?&1? @Iy?&1? @a??,S@?)?i????????Unknown
v@HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??v?? @9??v?? @A??v?? @I??v?? @a??z??(?i???nw????Unknown
tAHostReadVariableOp"Adam/Cast/ReadVariableOp(1????S??9????S??A????S??I????S??aVK???a#?i???????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1???S???9???S???A???S???I???S???a?-?g?G"?i݂j	?????Unknown
TCHostAbs"Abs(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a? ??B?!?i}_?}?????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_4(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a??
?-!?iZX????Unknown
oEHostReadVariableOp"Adam/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????ayI??-??i???i ????Unknown
?FHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????ayI??-??i??9{?????Unknown
tGHostAssignAddVariableOp"AssignAddVariableOp(1????x???9????x???A????x???I????x???a?Ou????iL%w
?????Unknown
VHHostSum"Sum_3(1????x???9????x???A????x???I????x???a?Ou????i?????????Unknown
?IHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1d;?O????9d;?O????Ad;?O????Id;?O????a`{???t?ir??=?????Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_5(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a??Æ?y?i???????Unknown
aKHostIdentity"Identity(1P??n???9P??n???AP??n???IP??n???ao??̏E?i^1?8X????Unknown?
?LHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?V-??9?V-??A?V-??I?V-??a?????i^?G????Unknown
?MHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??a?{,?i??Ĩ?????Unknown
?NHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1X9??v??9X9??v??AX9??v??IX9??v??a<F"?P?i??!)?????Unknown
{OHostSum"*categorical_crossentropy/weighted_loss/Sum(1??/?$??9??/?$??A??/?$??I??/?$??a\?[?il ??Q????Unknown
vPHostAssignAddVariableOp"AssignAddVariableOp_1(1?n?????9?n?????A?n?????I?n?????a????e??iZ?'????Unknown
bQHostDivNoNan"div_no_nan_1(1??C?l???9??C?l???A??C?l???I??C?l???a
??e?Z?i??b??????Unknown
XRHostCast"Cast_1(1/?$????9/?$????A/?$????I/?$????a?
???ir??[????Unknown
vSHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????x???9????x???A????x???I????x???a????0??i??? ????Unknown
TTHostMul"Mul(1?/?$??9?/?$??A?/?$??I?/?$??a????R&?i?6??????Unknown
`UHostDivNoNan"
div_no_nan(1'1?Z??9'1?Z??A'1?Z??I'1?Z??ab4\Ĭ??i???.????Unknown
bVHostDivNoNan"div_no_nan_2(11?Zd??91?Zd??A1?Zd??I1?Zd??a?u1???i?v?????Unknown
wWHostReadVariableOp"div_no_nan_2/ReadVariableOp(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a???k??iE &?9????Unknown
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a??&?
j?i?pQN?????Unknown
XYHostCast"Cast_3(1ˡE?????9ˡE?????AˡE?????IˡE?????a?U`?
?i)??T????Unknown
uZHostReadVariableOp"div_no_nan/ReadVariableOp(1ˡE?????9ˡE?????AˡE?????IˡE?????a?U`?
?irq\[?????Unknown
w[HostReadVariableOp"div_no_nan/ReadVariableOp_1(1ˡE?????9ˡE?????AˡE?????IˡE?????a?U`?
?i???a?????Unknown
y\HostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1ˡE?????9ˡE?????AˡE?????IˡE?????a?U`?
?irghS????Unknown
w]HostReadVariableOp"div_no_nan_1/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???aʇ????i*?O??????Unknown
?^HostDivNoNan",categorical_crossentropy/weighted_loss/value(1?z?G???9?z?G???A?z?G???I?z?G???a?Դ???i?????????Unknown*?W
uHostFlushSummaryWriter"FlushSummaryWriter(1%???J?@9%???J?@A%???J?@I%???J?@a?e????i?e?????Unknown?
?Host
_MklMatMul")gradient_tape/sequential/dense_1/MatMul_1(1'1??o@9'1??o@A'1??o@I'1??o@aT IQ%б?i&ܷ$?u???Unknown
?HostRandomStandardNormal"<sequential/gaussian_noise/random_normal/RandomStandardNormal(1u?V?k@9u?V?k@Au?V?k@Iu?V?k@a?#?s6L??iQ?y?????Unknown
?HostRandomStandardNormal">sequential/gaussian_noise_1/random_normal/RandomStandardNormal(1d;?O??j@9d;?O??j@Ad;?O??j@Id;?O??j@a??????i?~UH?????Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_1/MatMul(1?~j?t?f@9?~j?t?f@A?~j?t?f@I?~j?t?f@a?wi????i???~?+???Unknown
}Host
_MklMatMul"%gradient_tape/sequential/dense/MatMul(1??Q??f@9??Q??f@A??Q??f@I??Q??f@a?Q? )g??i???*????Unknown
zHost_MklNativeFusedMatMul"sequential/dense_1/Relu(17?A`??d@97?A`??d@A7?A`??d@I7?A`??d@a?x?K???is??N:???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1P??nsd@9P??nsd@AP??nsd@IP??nsd@a_???????i9??YY????Unknown
x	Host_MklNativeFusedMatMul"sequential/dense/Relu(1??C?lo]@9??C?lo]@A??C?lo]@I??C?lo]@aϡw????i*?o??????Unknown
}
Host_MklNativeFusedMatMul"sequential/dense_2/BiasAdd(1u?V~P@9u?V~P@Au?V~P@Iu?V~P@a?3?Ɣ???i???g?E???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1+??N@9+??N@A+??N@I+??N@ai-?jd_??i1????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1V-bG@9V-bG@AV-bG@IV-bG@an??=;??i????o9???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1????MRF@9????MRF@A????MRF@I????MRF@a $B
??iCws??????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1#??~j|@@9#??~j|@@A#??~j|@@I#??~j|@@a???dm~??i?aA?????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1J+??@9J+??@AJ+??@IJ+??@aƗ???o??iO$??Q-???Unknown
iHostWriteSummary"WriteSummary(1????K?;@9????K?;@A????K?;@I????K?;@a,c?ً?~?iyx?8k???Unknown?
?Host
_MklMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?"??~J;@9?"??~J;@A?"??~J;@I?"??~J;@a???d?~?iC???s????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??ʡE9@9??ʡE9@A??ʡE9@I??ʡE9@a???ps$|?i	?h??????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1D?l??i<@9D?l??i<@AZd;߯8@IZd;߯8@aX?eX??{?i?x????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#??~j<8@9#??~j<8@A#??~j<8@I#??~j<8@an???0{?i????N???Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1?|?5^Z7@9?|?5^Z7@A?|?5^Z7@I?|?5^Z7@a?a??{2z?i??)??????Unknown
^HostGatherV2"GatherV2(1Zd;??3@9Zd;??3@AZd;??3@IZd;??3@aU??{]v?id]ݟ????Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_2/MatMul(1Zd;??3@9Zd;??3@AZd;??3@IZd;??3@aU??{]v?iC>??Z????Unknown
oHostSoftmax"sequential/dense_2/Softmax(1d;?O??3@9d;?O??3@Ad;?O??3@Id;?O??3@aL??2Mv?i???m????Unknown
cHostDataset"Iterator::Root(1?z?G?F@9?z?G?F@A??Mb?,@I??Mb?,@aƇ??	p?i?P?(???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?ʡE??0@9?ʡE??0@A?O??n?&@I?O??n?&@a?????i?i???EB???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?C?l?;"@9?C?l?;"@A?C?l?;"@I?C?l?;"@a7?Pejtd?i?7Cl?V???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(11?Zd!@91?Zd!@A1?Zd!@I1?Zd!@aזPv??c?i3??<j???Unknown
eHost
LogicalAnd"
LogicalAnd(1'1?Z!@9'1?Z!@A'1?Z!@I'1?Z!@a3??wc?iH;z(?}???Unknown?
`HostGatherV2"
GatherV2_1(1???x?? @9???x?? @A???x?? @I???x?? @av?nV?b?i ??Aa????Unknown
[HostAddV2"Adam/add(133333s @933333s @A33333s @I33333s @a^?F?tb?i???Xբ???Unknown
? HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??ʡE @9??ʡE @A??ʡE @I??ʡE @a?h??@b?i?+Q????Unknown
g!HostStridedSlice"strided_slice(1     @ @9     @ @A     @ @I     @ @a(?25?:b?i?9`?P????Unknown
?"HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1+???@9+???@A+???@I+???@a???v?^?i?PZ????Unknown
Z#HostArgMax"ArgMax(1?n??J@9?n??J@A?n??J@I?n??J@a>ĵO?~]?i?ZC?????Unknown
V$HostMean"Mean(1T㥛? @9T㥛? @AT㥛? @IT㥛? @aں??90\?i&??1????Unknown
l%HostIteratorGetNext"IteratorGetNext(15^?I@95^?I@A5^?I@I5^?I@a?5܉?\?iA?ݎ8???Unknown
?&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1R???Q@9R???Q@AR???Q@IR???Q@a^?P??	Y?i?K?x????Unknown
?'HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1D?l??)@9D?l??)@AD?l??)@ID?l??)@a???X?i?X?+???Unknown
w(HostDataset""Iterator::Root::ParallelMapV2::Zip(1?z?G?J@9?z?G?J@A??(\??@I??(\??@a`}??iX?i>'P}`&???Unknown
?)HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1?t??@9?t??@A?t??@I?t??@a?+?L!U?iTwY#?0???Unknown
V*HostSum"Sum_2(1?ʡE??@9?ʡE??@A?ʡE??@I?ʡE??@a֭?dn#T?i+???;???Unknown
?+HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1d;?O??@9d;?O??@Ad;?O??@Id;?O??@ad?S?i; ???D???Unknown
|,HostMul"+sequential/gaussian_noise/random_normal/mul(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a?:S?i??}nN???Unknown
?-HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(17?A`??@97?A`????A7?A`??@I7?A`????aB?hB?P?i???V???Unknown
p.HostAddV2"sequential/gaussian_noise/add(1^?I+@9^?I+@A^?I+@I^?I+@a[p??6\P?i?z:?^???Unknown
[/HostPow"
Adam/Pow_1(1V-???@9V-???@AV-???@IV-???@a2E=??:P?i??l?g???Unknown
r0HostAddV2"sequential/gaussian_noise_1/add(1V-?
@9V-?
@AV-?
@IV-?
@am.z??M?i?N?n???Unknown
~1HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a?Y(K?i?~!P]u???Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a???J?i?D??|???Unknown
Y3HostPow"Adam/Pow(1)\???(@9)\???(@A)\???(@I)\???(@a?6<??I?i?s??????Unknown
~4HostMul"-sequential/gaussian_noise_1/random_normal/mul(1ˡE???@9ˡE???@AˡE???@IˡE???@a?*?Ot?G?i????}????Unknown
\5HostArgMax"ArgMax_1(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@a¶>m{F?ibOby????Unknown
]6HostCast"Adam/Cast_1(1???Q?@9???Q?@A???Q?@I???Q?@a??R?)F?id?C?????Unknown
?7HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1-???g3@9-???g3@Ao??ʡ@Io??ʡ@a???0?F?iI?޼%????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_6(1R???Q@9R???Q@AR???Q@IR???Q@a3?+?J?E?i/??ϐ????Unknown
?9HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1?G?z@9?G?z@A?G?z@I?G?z@aX?zf^gE?i?>"??????Unknown
T:HostSub"sub(1??Q?@9??Q?@A??Q?@I??Q?@a?g?_?SD?im,z??????Unknown
X;HostEqual"Equal(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a?(?g:D?i?6n,????Unknown
?<HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1Zd;?O?@9Zd;?O?@AZd;?O?@IZd;?O?@ai??L??C?i@hAP?????Unknown
V=HostCast"Cast(1??n?? @9??n?? @A??n?? @I??n?? @a?|x??B?i_???????Unknown
X>HostCast"Cast_2(1y?&1? @9y?&1? @Ay?&1? @Iy?&1? @aHy=?׳B?i??t?Y????Unknown
v?HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??v?? @9??v?? @A??v?? @I??v?? @aîM??B?i)i?(?????Unknown
t@HostReadVariableOp"Adam/Cast/ReadVariableOp(1????S??9????S??A????S??I????S??aC?6?i<?i1B^k????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1???S???9???S???A???S???I???S???a????:?ir????????Unknown
TBHostAbs"Abs(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a??$
?:?iz?????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1??? ?r??9??? ?r??A??? ?r??I??? ?r??a??{"?.9?i}j>?,????Unknown
oDHostReadVariableOp"Adam/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????a$:y?^7?im|????Unknown
?EHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?E??????9?E??????A?E??????I?E??????a$:y?^7?i??L????Unknown
tFHostAssignAddVariableOp"AssignAddVariableOp(1????x???9????x???A????x???I????x???a?ձk75?iB/?0?????Unknown
VGHostSum"Sum_3(1????x???9????x???A????x???I????x???a?ձk75?i}?wR????Unknown
?HHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1d;?O????9d;?O????Ad;?O????Id;?O????a?ž9?4?i6~?{?????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_5(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a֭?dn#4?i?|?q????Unknown
aJHostIdentity"Identity(1P??n???9P??n???AP??n???IP??n???a?r????2?i?t??????Unknown?
?KHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1?V-??9?V-??A?V-??I?V-??a??5?f%2?i??R????Unknown
?LHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??a?Qy?&?1?i?j?W;????Unknown
?MHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1X9??v??9X9??v??AX9??v??IX9??v??a)>'N1?i?M?!^????Unknown
{NHostSum"*categorical_crossentropy/weighted_loss/Sum(1??/?$??9??/?$??A??/?$??I??/?$??a?7?P[?0?i?a?,{????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_1(1?n?????9?n?????A?n?????I?n?????a?Ʌ1е0?i?????????Unknown
bPHostDivNoNan"div_no_nan_1(1??C?l???9??C?l???A??C?l???I??C?l???a8??nM/?im?콆????Unknown
XQHostCast"Cast_1(1/?$????9/?$????A/?$????I/?$????a?9??.?i????u????Unknown
vRHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????x???9????x???A????x???I????x???a??N?0.?i?U?X????Unknown
TSHostMul"Mul(1?/?$??9?/?$??A?/?$??I?/?$??a????[,?i??????Unknown
`THostDivNoNan"
div_no_nan(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a??^?_Q+?i?'?????Unknown
bUHostDivNoNan"div_no_nan_2(11?Zd??91?Zd??A1?Zd??I1?Zd??a-????=*?i-1??r????Unknown
wVHostReadVariableOp"div_no_nan_2/ReadVariableOp(1??x?&1??9??x?&1??A??x?&1??I??x?&1??a?	b??&?iN'aa?????Unknown
yWHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a??ur?J&?i?NhB????Unknown
XXHostCast"Cast_3(1ˡE?????9ˡE?????AˡE?????IˡE?????al?)^?#?iF1>
s????Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1ˡE?????9ˡE?????AˡE?????IˡE?????al?)^?#?i??????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1ˡE?????9ˡE?????AˡE?????IˡE?????al?)^?#?iv???????Unknown
y[HostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1ˡE?????9ˡE?????AˡE?????IˡE?????al?)^?#?iٿ?????Unknown
w\HostReadVariableOp"div_no_nan_1/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???au*??u? ?ia?????Unknown
?]HostDivNoNan",categorical_crossentropy/weighted_loss/value(1?z?G???9?z?G???A?z?G???I?z?G???a?Ot?\'?i     ???Unknown2CPU