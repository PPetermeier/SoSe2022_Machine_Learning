"?W
BHostIDLE"IDLE1??/}??@A??/}??@a^??_???i^??_????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1y?&1?
?@9y?&1?
?@Ay?&1?
?@Iy?&1?
?@a????v ??iJr?2????Unknown?
?HostRandomStandardNormal"<sequential/gaussian_noise/random_normal/RandomStandardNormal(1?V?k@9?V?k@A?V?k@I?V?k@a??z_F???i@J?;?O???Unknown
xHost_MklNativeFusedMatMul"sequential/dense/Relu(1??/??X@9??/??X@A??/??X@I??/??X@a%%&z????i???!?????Unknown
zHost_MklNativeFusedMatMul"sequential/dense_1/Relu(1????x?W@9????x?W@A????x?W@I????x?W@ay?L]???i/?????Unknown
?HostRandomStandardNormal">sequential/gaussian_noise_1/random_normal/RandomStandardNormal(1P??n?W@9P??n?W@AP??n?W@IP??n?W@a??y??iN?;?yD???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??/݌U@9??/݌U@A??/݌U@I??/݌U@a<??ո8??ic??r\????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?ʡE?Q@9?ʡE?Q@A?ʡE?Q@I?ʡE?Q@a[?\???|?il??#?????Unknown
	Host
_MklMatMul"'gradient_tape/sequential/dense_1/MatMul(1+?9L@9+?9L@A+?9L@I+?9L@a?Tm?w?i2Z?R?????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1?G?z?F@9?G?z?F@A?G?z?F@I?G?z?F@a?>F??s?i????????Unknown
?Host
_MklMatMul")gradient_tape/sequential/dense_1/MatMul_1(1ˡE??E@9ˡE??E@AˡE??E@IˡE??E@ar䱺i?q?ixJL?|@???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1=
ףp?C@9=
ףp?C@A=
ףp?C@I=
ףp?C@aL??w?p?i?cY?/b???Unknown
^HostGatherV2"GatherV2(1?S㥛?B@9?S㥛?B@A?S㥛?B@I?S㥛?B@a?'$_p?i,??=????Unknown
}Host_MklNativeFusedMatMul"sequential/dense_2/BiasAdd(1?K7?A?@@9?K7?A?@@A?K7?A?@@I?K7?A?@@a???9??k?i?mQ;%????Unknown
}Host
_MklMatMul"%gradient_tape/sequential/dense/MatMul(1? ?rh?@9? ?rh?@A? ?rh?@I? ?rh?@a???Dj?i?8^%j????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1?&1?|>@9?&1?|>@A?&1?|>@I?&1?|>@a??y?&?i?i??UL1????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1d;?O??;@9d;?O??;@Ad;?O??;@Id;?O??;@a?'???g?i??R??????Unknown
iHostWriteSummary"WriteSummary(1B`??"{;@9B`??"{;@AB`??"{;@IB`??"{;@a????z<g?i?[#f? ???Unknown?
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1)\???8@9)\???8@A?l????4@I?l????4@a'??Rda?i?=??\???Unknown
oHostSoftmax"sequential/dense_2/Softmax(1?? ?r?1@9?? ?r?1@A?? ?r?1@I?? ?r?1@a?????^?iR ?e!???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1J+??1@9J+??1@AJ+??1@IJ+??1@a????<?]?i?ȍ`0???Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_2/MatMul(1????x)'@9????x)'@A????x)'@I????x)'@aǨl??S?i?~?*:???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1????K?0@9????K?0@A??x?&?%@I??x?&?%@a??<???R?i@??qC???Unknown
?Host
_MklMatMul")gradient_tape/sequential/dense_2/MatMul_1(1sh??|$@9sh??|$@Ash??|$@Ish??|$@a??E?TQ?i???%L???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1L7?A`e"@9L7?A`e"@AL7?A`e"@IL7?A`e"@a??kX?O?i?ǎ ?S???Unknown
cHostDataset"Iterator::Root(1ףp=
?:@9ףp=
?:@A/?$A"@I/?$A"@aL?]?q?N?i???[???Unknown
`HostGatherV2"
GatherV2_1(1???Mb?!@9???Mb?!@A???Mb?!@I???Mb?!@a?7Z?N?iƬ??"c???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?n??J!@9?n??J!@A?n??J!@I?n??J!@a?B??=M?iK=a9rj???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1H?z?? @9H?z?? @AH?z?? @IH?z?? @a?)O1?K?i???oq???Unknown
gHostStridedSlice"strided_slice(1=
ףp} @9=
ףp} @A=
ףp} @I=
ףp} @a?-z"??K?i?/??gx???Unknown
eHost
LogicalAnd"
LogicalAnd(1F????x @9F????x @AF????x @IF????x @a??3B??K?i???^???Unknown?
? HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?z?G! @9?z?G! @A?z?G! @I?z?G! @aqU???FK?i???l0????Unknown
?!HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1V-??@9V-??@AV-??@IV-??@a??d?
I?i1s????Unknown
Z"HostArgMax"ArgMax(1?????K@9?????K@A?????K@I?????K@a??Y?H?iIY?j?????Unknown
[#HostAddV2"Adam/add(11?Z?@91?Z?@A1?Z?@I1?Z?@aJ????mH?i-J]俘???Unknown
$HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1Zd;??@9Zd;??@AZd;??@IZd;??@as "^??F?i??4?j????Unknown
l%HostIteratorGetNext"IteratorGetNext(1/?$?@9/?$?@A/?$?@I/?$?@a?)޶?C?i ]??K????Unknown
w&HostDataset""Iterator::Root::ParallelMapV2::Zip(1?z?G!H@9?z?G!H@A???Q?@I???Q?@a	!Gt?5C?i?n	x????Unknown
?'HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a5.?G?C?i?`[?٬???Unknown
V(HostMean"Mean(1?~j?t@9?~j?t@A?~j?t@I?~j?t@a&r????@?i?LT]????Unknown
?)HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?E????@9?E????@A?E????@I?E????@acȰ??v<?i?B-8?????Unknown
|*HostMul"+sequential/gaussian_noise/random_normal/mul(1??n??@9??n??@A??n??@I??n??@a*??4T<?i??¾1????Unknown
p+HostAddV2"sequential/gaussian_noise/add(1?|?5^:@9?|?5^:@A?|?5^:@I?|?5^:@a????[q;?i5?5ꟻ???Unknown
?,HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1?&1?@9?&1?@A?&1?@I?&1?@aj??#?;?ix??????Unknown
V-HostSum"Sum_2(17?A`??@97?A`??@A7?A`??@I7?A`??@aϭE$??:?i? ?d????Unknown
Y.HostPow"Adam/Pow(1#??~j?@9#??~j?@A#??~j?@I#??~j?@a???/?s7?i???S????Unknown
?/HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?|?5^?@9?|?5^???A?|?5^?@I?|?5^???a-???q7?i?-?\A????Unknown
?0HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a?(??
7?i??e?"????Unknown
X1HostCast"Cast_2(1333333@9333333@A333333@I333333@a?O?Hv4?i1fg?????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1???Q?@9???Q?@A???Q?@I???Q?@a???:w]2?iLqM?????Unknown
\3HostArgMax"ArgMax_1(1??????@9??????@A??????@I??????@aLA??ݽ0?itN?????Unknown
]4HostCast"Adam/Cast_1(1?Zd;?@9?Zd;?@A?Zd;?@I?Zd;?@a:VRI9.?i??,c?????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_6(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a???y?F-?ib~d??????Unknown
[6HostPow"
Adam/Pow_1(1??n?? @9??n?? @A??n?? @I??n?? @a*??4T,?i???????Unknown
V7HostCast"Cast(1'1?Z @9'1?Z @A'1?Z @I'1?Z @a??
?+?izP?L????Unknown
~8HostMul"-sequential/gaussian_noise_1/random_normal/mul(1?&1? @9?&1? @A?&1? @I?&1? @aj??#?+?i?E?s ????Unknown
T9HostSub"sub(1??C?l??9??C?l??A??C?l??I??C?l??aSCe(??*?i????????Unknown
?:HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1?Q???2@9?Q???2@A?$??C??I?$??C??aL?
Wo*?i?w??P????Unknown
t;HostReadVariableOp"Adam/Cast/ReadVariableOp(1??v????9??v????A??v????I??v????a?T??L*?i?H?S?????Unknown
?<HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1??S㥛??9??S㥛??A??S㥛??I??S㥛??ap??W?)?i??]i?????Unknown
?=HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a??.??)?i???*-????Unknown
T>HostAbs"Abs(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a}ي??A(?ik??F?????Unknown
X?HostEqual"Equal(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a?6Xz?'?i??(1????Unknown
~@HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a???:??'?i?諫?????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_5(1?5^?I??9?5^?I??A?5^?I??I?5^?I??a&?I7aU$?iG]??????Unknown
?BHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1???S???9???S???A???S???I???S???a????2$?iE??-7????Unknown
oCHostReadVariableOp"Adam/ReadVariableOp(1+?????9+?????A+?????I+?????aS???p"?i*w?5^????Unknown
rDHostAddV2"sequential/gaussian_noise_1/add(1ˡE?????9ˡE?????AˡE?????IˡE?????a?g??!?i??4z????Unknown
tEHostAssignAddVariableOp"AssignAddVariableOp(1?E??????9?E??????A?E??????I?E??????a??qI@?!?i??????Unknown
vFHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a??Zo.!?i?u?j?????Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_4(1???S???9???S???A???S???I???S???ax?4?? ?i??y?????Unknown
?HHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1J+???9J+???AJ+???IJ+???a`负?# ?ia????????Unknown
vIHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1??n????9??n????A??n????I??n????a????	??ia.?e?????Unknown
VJHostSum"Sum_3(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a?(?*??i??6ϩ????Unknown
aKHostIdentity"Identity(1?&1???9?&1???A?&1???I?&1???a?;P????i??֞????Unknown?
vLHostAssignAddVariableOp"AssignAddVariableOp_1(1F????x??9F????x??AF????x??IF????x??a#? ????i?96?????Unknown
{MHostSum"*categorical_crossentropy/weighted_loss/Sum(1ˡE?????9ˡE?????AˡE?????IˡE?????aӶ??i>?q????Unknown
?NHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1ˡE?????9ˡE?????AˡE?????IˡE?????aӶ??id???V????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_3(1P??n???9P??n???AP??n???IP??n???a??QN??i?W\Z6????Unknown
?PHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a??
??iS??????Unknown
TQHostMul"Mul(1m???????9m???????Am???????Im???????a$????7?i`?O?????Unknown
?RHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?Zd;??9?Zd;??A?Zd;??I?Zd;??af) ???iact??????Unknown
XSHostCast"Cast_1(19??v????99??v????A9??v????I9??v????a?Jz????i3?A?I????Unknown
bTHostDivNoNan"div_no_nan_1(1?C?l????9?C?l????A?C?l????I?C?l????a?m?7q??iƕ?_?????Unknown
bUHostDivNoNan"div_no_nan_2(1???Mb??9???Mb??A???Mb??I???Mb??aE????X?i?4?&?????Unknown
yVHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1??v????9??v????A??v????I??v????a??!??i?C?n8????Unknown
yWHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1F????x??9F????x??AF????x??IF????x??a?e??'?i`D)??????Unknown
wXHostReadVariableOp"div_no_nan/ReadVariableOp_1(1??C?l???9??C?l???A??C?l???I??C?l???a????`??i?/PP????Unknown
`YHostDivNoNan"
div_no_nan(1?$??C??9?$??C??A?$??C??I?$??C??a???Y?I?i?????????Unknown
uZHostReadVariableOp"div_no_nan/ReadVariableOp(1??ʡE??9??ʡE??A??ʡE??I??ʡE??aQ*??<??i.?7N????Unknown
w[HostReadVariableOp"div_no_nan_2/ReadVariableOp(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??a?pT?+??i?Е|?????Unknown
X\HostCast"Cast_3(1y?&1???9y?&1???Ay?&1???Iy?&1???a??\??1?i??B6????Unknown
w]HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??aSCe(??
?i??Ċ?????Unknown
?^HostDivNoNan",categorical_crossentropy/weighted_loss/value(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a???N??i      ???Unknown*?W
uHostFlushSummaryWriter"FlushSummaryWriter(1y?&1?
?@9y?&1?
?@Ay?&1?
?@Iy?&1?
?@a?1?@[]??i?1?@[]???Unknown?
?HostRandomStandardNormal"<sequential/gaussian_noise/random_normal/RandomStandardNormal(1?V?k@9?V?k@A?V?k@I?V?k@a??VVxѰ?i)?K?w???Unknown
xHost_MklNativeFusedMatMul"sequential/dense/Relu(1??/??X@9??/??X@A??/??X@I??/??X@a?(M/????inq??Yg???Unknown
zHost_MklNativeFusedMatMul"sequential/dense_1/Relu(1????x?W@9????x?W@A????x?W@I????x?W@a?&aR#???i?z??M???Unknown
?HostRandomStandardNormal">sequential/gaussian_noise_1/random_normal/RandomStandardNormal(1P??n?W@9P??n?W@AP??n?W@IP??n?W@a߱ wi???i2?F#?2???Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1??/݌U@9??/݌U@A??/݌U@I??/݌U@a Rv???i?J?F????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?ʡE?Q@9?ʡE?Q@A?ʡE?Q@I?ʡE?Q@a$?h?)???i?????????Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_1/MatMul(1+?9L@9+?9L@A+?9L@I+?9L@aW? 	??i?F/???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1?G?z?F@9?G?z?F@A?G?z?F@I?G?z?F@a?ܒ,?M??iz]??}????Unknown
?
Host
_MklMatMul")gradient_tape/sequential/dense_1/MatMul_1(1ˡE??E@9ˡE??E@AˡE??E@IˡE??E@a7?厯j??i[??Y(???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1=
ףp?C@9=
ףp?C@A=
ףp?C@I=
ףp?C@aj?n?e??iɮ?ab???Unknown
^HostGatherV2"GatherV2(1?S㥛?B@9?S㥛?B@A?S㥛?B@I?S㥛?B@a???????i?????????Unknown
}Host_MklNativeFusedMatMul"sequential/dense_2/BiasAdd(1?K7?A?@@9?K7?A?@@A?K7?A?@@I?K7?A?@@a*3??x???iyj?ޗ???Unknown
}Host
_MklMatMul"%gradient_tape/sequential/dense/MatMul(1? ?rh?@9? ?rh?@A? ?rh?@I? ?rh?@a"?F0????i??Lc?X???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1?&1?|>@9?&1?|>@A?&1?|>@I?&1?|>@a⪭}?f??i^<C?5????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1d;?O??;@9d;?O??;@Ad;?O??;@Id;?O??;@a????%΀?i^??dn????Unknown
iHostWriteSummary"WriteSummary(1B`??"{;@9B`??"{;@AB`??"{;@IB`??"{;@a	kN?]???i
ٟ??'???Unknown?
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1)\???8@9)\???8@A?l????4@I?l????4@a m????x?i???&qY???Unknown
oHostSoftmax"sequential/dense_2/Softmax(1?? ?r?1@9?? ?r?1@A?? ?r?1@I?? ?r?1@a???2?wu?i??,y`????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1J+??1@9J+??1@AJ+??1@IJ+??1@a??3bu?iK???$????Unknown
Host
_MklMatMul"'gradient_tape/sequential/dense_2/MatMul(1????x)'@9????x)'@A????x)'@I????x)'@a??????k?i?x?????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1????K?0@9????K?0@A??x?&?%@I??x?&?%@a???z?|j?i???y?????Unknown
?Host
_MklMatMul")gradient_tape/sequential/dense_2/MatMul_1(1sh??|$@9sh??|$@Ash??|$@Ish??|$@aB@sv??h?i4+j.V????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1L7?A`e"@9L7?A`e"@AL7?A`e"@IL7?A`e"@a?[J@5f?i?u?H????Unknown
cHostDataset"Iterator::Root(1ףp=
?:@9ףp=
?:@A/?$A"@I/?$A"@a&8	f?i?????*???Unknown
`HostGatherV2"
GatherV2_1(1???Mb?!@9???Mb?!@A???Mb?!@I???Mb?!@a??ֽ=?e?i[^n?@???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?n??J!@9?n??J!@A?n??J!@I?n??J!@a????d?iv???T???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1H?z?? @9H?z?? @AH?z?? @IH?z?? @aF?So?c?iOc??h???Unknown
gHostStridedSlice"strided_slice(1=
ףp} @9=
ףp} @A=
ףp} @I=
ףp} @aT?p?c?i:??+?|???Unknown
eHost
LogicalAnd"
LogicalAnd(1F????x @9F????x @AF????x @IF????x @a??a??c?i?Y??????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?z?G! @9?z?G! @A?z?G! @I?z?G! @aۍLL?xc?iUX?-????Unknown
? HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1V-??@9V-??@AV-??@IV-??@aI9D?a?i?p??????Unknown
Z!HostArgMax"ArgMax(1?????K@9?????K@A?????K@I?????K@aT?H?Ѯa?i/?Ε?????Unknown
["HostAddV2"Adam/add(11?Z?@91?Z?@A1?Z?@I1?Z?@a?R??dpa?i?{??,????Unknown
#HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1Zd;??@9Zd;??@AZd;??@IZd;??@at(|?.`?i????[????Unknown
l$HostIteratorGetNext"IteratorGetNext(1/?$?@9/?$?@A/?$?@I/?$?@a?}???[?i????J????Unknown
w%HostDataset""Iterator::Root::ParallelMapV2::Zip(1?z?G!H@9?z?G!H@A???Q?@I???Q?@a??7cdm[?i??ڛ???Unknown
?&HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a[ ?8#[?i?U8????Unknown
V'HostMean"Mean(1?~j?t@9?~j?t@A?~j?t@I?~j?t@a-e?<J<X?i?t]????Unknown
?(HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?E????@9?E????@A?E????@I?E????@a??[??QT?i??6??(???Unknown
|)HostMul"+sequential/gaussian_noise/random_normal/mul(1??n??@9??n??@A??n??@I??n??@a??`
9T?i?+g??2???Unknown
p*HostAddV2"sequential/gaussian_noise/add(1?|?5^:@9?|?5^:@A?|?5^:@I?|?5^:@a?K?S?i?ѦQ?<???Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1?&1?@9?&1?@A?&1?@I?&1?@a?/?=AsS?i9?E?{F???Unknown
V,HostSum"Sum_2(17?A`??@97?A`??@A7?A`??@I7?A`??@a|X??54S?ie?<P???Unknown
Y-HostPow"Adam/Pow(1#??~j?@9#??~j?@A#??~j?@I#??~j?@ay?S?ĽP?i^,??tX???Unknown
?.HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?|?5^?@9?|?5^???A?|?5^?@I?|?5^???a?NX??P?i?S?3?`???Unknown
?/HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1sh??|?@9sh??|?@Ash??|?@Ish??|?@ao,?\rP?iT?Dbi???Unknown
X0HostCast"Cast_2(1333333@9333333@A333333@I333333@a?R???6M?i??Zp???Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1???Q?@9???Q?@A???Q?@I???Q?@a*?%Z8J?i-*?v???Unknown
\2HostArgMax"ArgMax_1(1??????@9??????@A??????@I??????@a?i???G?i[????|???Unknown
]3HostCast"Adam/Cast_1(1?Zd;?@9?Zd;?@A?Zd;?@I?Zd;?@a?^*?E?i?Yq?F????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_6(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a??_?D?i?L	;?????Unknown
[5HostPow"
Adam/Pow_1(1??n?? @9??n?? @A??n?? @I??n?? @a??`
9D?i???}?????Unknown
V6HostCast"Cast(1'1?Z @9'1?Z @A'1?Z @I'1?Z @a<???l?C?i?D??}????Unknown
~7HostMul"-sequential/gaussian_noise_1/random_normal/mul(1?&1? @9?&1? @A?&1? @I?&1? @a?/?=AsC?i%??Z????Unknown
T8HostSub"sub(1??C?l??9??C?l??A??C?l??I??C?l??a,䭇??B?i????????Unknown
?9HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1?Q???2@9?Q???2@A?$??C??I?$??C??aJFc??B?i o?LП???Unknown
t:HostReadVariableOp"Adam/Cast/ReadVariableOp(1??v????9??v????A??v????I??v????ah,?>1?B?i?&?؁????Unknown
?;HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1??S㥛??9??S㥛??A??S㥛??I??S㥛??a?5???yB?i? < ????Unknown
?<HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a4~̞HB?i???B?????Unknown
T=HostAbs"Abs(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a`??2?PA?iRp?z????Unknown
X>HostEqual"Equal(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a?/??lA?i???UN????Unknown
~?HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a?S?ųA?isM??????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_5(1?5^?I??9?5^?I??A?5^?I??I?5^?I??a???%?=?i?E?0????Unknown
?AHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1???S???9???S???A???S???I???S???a?E$?Y?<?i|????????Unknown
oBHostReadVariableOp"Adam/ReadVariableOp(1+?????9+?????A+?????I+?????a???*?S:?iz9????Unknown
rCHostAddV2"sequential/gaussian_noise_1/add(1ˡE?????9ˡE?????AˡE?????IˡE?????ao?t?^W9?iR?$A????Unknown
tDHostAssignAddVariableOp"AssignAddVariableOp(1?E??????9?E??????A?E??????I?E??????a?ܤ??%9?i??w?e????Unknown
vEHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a.?zRe8?i?!?r????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_4(1???S???9???S???A???S???I???S???a?S??08?ik?9?r????Unknown
?GHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1J+???9J+???AJ+???IJ+???aӼ?T?
7?i??)T????Unknown
vHHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1??n????9??n????A??n????I??n????a????6?in(????Unknown
VIHostSum"Sum_3(1}?5^?I??9}?5^?I??A}?5^?I??I}?5^?I??a?%???6?iم??????Unknown
aJHostIdentity"Identity(1?&1???9?&1???A?&1???I?&1???a????U?5?i???????Unknown?
vKHostAssignAddVariableOp"AssignAddVariableOp_1(1F????x??9F????x??AF????x??IF????x??a?ɚ??5?igN?I????Unknown
{LHostSum"*categorical_crossentropy/weighted_loss/Sum(1ˡE?????9ˡE?????AˡE?????IˡE?????aL?+?5?4?i???????Unknown
?MHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1ˡE?????9ˡE?????AˡE?????IˡE?????aL?+?5?4?iO??nj????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_3(1P??n???9P??n???AP??n???IP??n???a {????3?i??J?????Unknown
?OHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1'1?Z??9'1?Z??A'1?Z??I'1?Z??a<???l?3?ivt5?_????Unknown
TPHostMul"Mul(1m???????9m???????Am???????Im???????aXM?h 2?i`?8?????Unknown
?QHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a???o0?i?	??????Unknown
XRHostCast"Cast_1(19??v????99??v????A9??v????I9??v????a??C?0?i=z???????Unknown
bSHostDivNoNan"div_no_nan_1(1?C?l????9?C?l????A?C?l????I?C?l????aQ`?c^/?i??w!?????Unknown
bTHostDivNoNan"div_no_nan_2(1???Mb??9???Mb??A???Mb??I???Mb??a!???-?iWW?v????Unknown
yUHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1??v????9??v????A??v????I??v????a?t*E?+?iƩ[.5????Unknown
yVHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1F????x??9F????x??AF????x??IF????x??a???ĵ?)?i????????Unknown
wWHostReadVariableOp"div_no_nan/ReadVariableOp_1(1??C?l???9??C?l???A??C?l???I??C?l???a:??"(?iG?[T????Unknown
`XHostDivNoNan"
div_no_nan(1?$??C??9?$??C??A?$??C??I?$??C??a*:?qYA'?ibxq?????Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1??ʡE??9??ʡE??A??ʡE??I??ʡE??ak`??&?i???])????Unknown
wZHostReadVariableOp"div_no_nan_2/ReadVariableOp(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??a\?j??H%?iV???}????Unknown
X[HostCast"Cast_3(1y?&1???9y?&1???Ay?&1???Iy?&1???a?2?<Q $?i????????Unknown
w\HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??a,䭇??"?i?5m?????Unknown
?]HostDivNoNan",categorical_crossentropy/weighted_loss/value(1?MbX9??9?MbX9??A?MbX9??I?MbX9??a????,	!?i     ???Unknown2CPU