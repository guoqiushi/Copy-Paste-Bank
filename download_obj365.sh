mkdir Objects365-2020/
mkdir Objects365-2020/license/
mkdir Objects365-2020/train/
mkdir -p Objects365-2020/val/images/v1/
mkdir -p Objects365-2020/val/images/v2/
mkdir -p Objects365-2020/test/images/v1/
mkdir -p Objects365-2020/test/images/v2/


cd Objects365-2020/

wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/license/license.txt.tar.gz -P license/
wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.json -P train/
wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json -P val/
wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/sample_2020.json.tar.gz -P val/

# train
for i in {0..50}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch${i}.tar.gz -P train/
done

# val
for i in {0..15}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/patch${i}.tar.gz -P val/images/v1
done

for i in {16..50}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/patch${i}.tar.gz -P val/images/v2/
done


# test
for i in {0..15}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/test/images/v1/patch${i}.tar.gz -P test/images/v1/
done

for i in {16..50}
  do wget -c https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/test/images/v2/patch${i}.tar.gz -P test/images/v2/
done
