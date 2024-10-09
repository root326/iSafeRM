# SafeScaler

## 1. 工作流程

1. 微服务运行过程中收集Jeager记录的trace数据，分析每一个span的运行延迟对端到端延迟的影响。我们将收集的trace数据根据SLO(200ms)标准分为正常数据与异常数据，利用因果推断与反事实推理，从正常数据分析因果机制，应用到异常数据中，找到每一个span对端到端延迟的因果贡献，并排序输出异常根因服务
2. 找到需要调整的服务后，为每一个服务选择合适的配置参数，形成一个配置空间，准备探索
3. 利用安全强化学习探索配置空间找到最优配置



## 2. 项目结构

1. .tmp文件夹用于存放微服务部署时的文件，部署过程中必要的配置文件、volumes等等会放入其中，服务删除后会自动清空
2. configs文件夹存放微服务调优所需要的一些静态的配置文件，csv格式，SafeScaler可以通过加载csv文件中的数据获取所需调整的参数
3. data文件夹下两个子目录，rca存放了一些历史捕获的trace数据，构建好的因果图，以及输出结果的根因服务json文件。replay下存放了不同配置下的运行数据，用于离线运行进行加载，不再在线部署获取数据。
4. deploy文件夹下存放微服务的一些部署文件，以ansible文件为主，在线部署时，会利用ansible执行任务。
5. output文件夹用于存放系统运行的输出结果
6. tuning文件夹存放SafeScaler具体的一些代码，包括指标采集，根因定位，强化学习。具体运行看safescaler.py

## 3. 离线运行

1. 由于离线运行只是加载收集的历史数据，在data/replay中存放了工作负载rps为10的环境下，safescaler的运行数据。启动时系统会加载csv中的数据，最终给出一个在历史数据上跑出的最优配置，存放于output/result/best.yml。result下的文件夹存放safescaler运行时的历史数据。
2. 关于根因定位，收集的历史数据存放于data/rca中,ms_data.csv是收集的trace数据，离线运行根因定位会加载其中的数据并给出结果放入top_services.json。由于根因定位根据采集的数据量与trace的大小会影响运行时间，所以方便起见可以直接拿准备好的top_services.json使用。
3. 此外，离线默认加载congfigs下的csv文件作为配置空间，所以csv中的数据可能会与top_services.json中的服务不同。

## 4. 在线运行

1. 在线运行默认不打开，需要导入microservices_env中的MicroservicesENV环境。
2. 在线运行一般先部署服务，根因定位，再加载信息进行训练，在safescaler.py中有具体步骤，当根因定位将服务写入top_services.json中后，在线运行会加载其中的服务并匹配配置参数形成配置空间，进行探索。在文件中默认没启动rca，直接选择已写好的top_services.json启动。
3. 定位与调优功能是分离的，因此在线运行也可以选择congfigs下的csv文件作为配置空间，具体可参考MicroservicesENVOffline中的配置初始化函数。

