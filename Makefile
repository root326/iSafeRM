SHELL := /bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: run-offline
run-offline:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safescaler.py -on no -rca no -ts 64 -spe 16 -bs 8 -ui 2

.PHONY: run-online
run-online:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safescaler.py -on yes -ts 64 -spe 16 -bs 8 -ui 2



.PHONY: deploy-playbook-socialnetwork
deploy-playbook-socialnetwork:
	@cd ~/iSafeRM/deploy/social_network/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook deploy_default.yml

.PHONY: clean-playbook-socialnetwork
clean-playbook-socialnetwork:
	@cd ~/iSafeRM/deploy/social_network/ansible && $(CONDA_ACTIVATE) activate mytunning && \
		ansible-playbook clean.yml


.PHONY: clean-playbook-hotel-reservation
clean-playbook-hotel-reservation:
	@cd ~/iSafeRM/deploy/hotel_reservation/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook clean.yml

.PHONY: deploy-playbook-hotel-reservation
deploy-playbook-hotel-reservation:
	@cd ~/iSafeRM/deploy/hotel_reservation/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook deploy.yml

.PHONY: deploy-playbook-media-microservices
deploy-playbook-media-microservices:
	@cd ~/iSafeRM/deploy/media_microservices/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook deploy.yml

.PHONY: clean-playbook-media-microservices
clean-playbook-media-microservices:
	@cd ~/iSafeRM/deploy/media_microservices/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook clean.yml

.PHONY: run
run:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn sn_both_all_bo_tpe_200_3_150 -b sn -it 200 -re 3 -para both -dr all -acq ei -algo bo_tpe

.PHONY: run-resource
run-resource:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn sn_resource_cp_bo_tpe_30_5_90 -b sn -it 30 -re 5 -para resource -dr critical_path -acq ei -algo bo_tpe


.PHONY: deploy-playbook-mongodb
deploy-playbook-mongodb:
	@cd ~/iSafeRM/deploy/software/mongodb/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook mongodb_deploy.yml

.PHONY: clean-playbook-mongodb
clean-playbook-mongodb:
	@cd ~/iSafeRM/deploy/software/mongodb/ansible && $(CONDA_ACTIVATE) activate mytunning && \
		ansible-playbook mongodb_clean.yml

.PHONY: wk-load-mongodb
wk-load-mongodb:
	@cd ~/iSafeRM/deploy/wk/ycsb-0.17.0	&& $(CONDA_ACTIVATE) activate mytunning && \
	  	./bin/ycsb.sh load mongodb-async -s -P ../mongodb/workloada > ../mongodb/workloada_load.txt

.PHONY: wk-run-mongodb
wk-run-mongodb:
	@cd ~/iSafeRM/deploy/wk/ycsb-0.17.0	&& $(CONDA_ACTIVATE) activate mytunning && \
	  	./bin/ycsb.sh run mongodb-async -s -P ../mongodb/workloada > ../mongodb/workloada_run.txt

.PHONY: run-mongodb
run-mongodb:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn mongodb_both_all_bo_tpe_2000 -b mongodb -it 2000 -re 1 -para both -dr all -acq ei -algo bo_tpe


.PHONY: deploy-playbook-prometheus
deploy-playbook-prometheus:
	@cd ~/iSafeRM/deploy/software/prometheus/ && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook ~/iSafeRM/deploy/software/prometheus/prometheus_deploy.yml


.PHONY: clean-playbook-prometheus
clean-playbook-prometheus:
	@cd ~/iSafeRM/deploy/software/prometheus/ && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook ~/iSafeRM/deploy/software/prometheus/prometheus_clean.yml


.PHONY: deploy-playbook-grafana
deploy-playbook-grafana:
	@cd /home/XXXX-1/iSafeRM/deploy/software/grafana/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook grafana_deploy.yml


.PHONY: clean-playbook-kibana
clean-playbook-kibana:
	@cd /home/XXXX-1/iSafeRM/deploy/software/kibana/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook kibana_clean.yml


.PHONY: deploy-playbook-kibana
deploy-playbook-kibana:
	@cd /home/XXXX-1/iSafeRM/deploy/software/kibana/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook kibana_deploy.yml


.PHONY: clean-playbook-grafana
clean-playbook-grafana:
	@cd /home/XXXX-1/iSafeRM/deploy/software/grafana/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook grafana_clean.yml


.PHONY: get_software_metric
get_software_metric:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python ~/iSafeRM/tunning/datacollector/metrics_collector.py --software_name mongodb


.PHONY: run-redis
run-redis:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn redis_both_all_bo_tpe_2000 -b redis -it 2000 -re 1 -para both -dr all -acq ei -algo bo_tpe


.PHONY: deploy-playbook-redis
deploy-playbook-redis:
	@cd ~/iSafeRM/deploy/software/redis/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook redis_deploy.yml

.PHONY: clean-playbook-redis
clean-playbook-redis:
	@cd ~/iSafeRM/deploy/software/redis/ansible && $(CONDA_ACTIVATE) activate mytunning && \
		ansible-playbook redis_clean.yml



.PHONY: run-nginx
run-nginx:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn nginx_both_all_bo_tpe_2000 -b nginx -it 2000 -re 1 -para both -dr all -acq ei -algo bo_tpe

.PHONY: run-prometheus
run-prometheus:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/datacollector/prometheus_deploy.py

.PHONY: run-memcached
run-memcached:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn memcached_both_all_bo_tpe_2000 -b memcached -it 4000 -re 1 -para both -dr all -acq ei -algo bo_tpe


.PHONY: deploy-playbook-mysql
deploy-playbook-mysql:
	@cd ~/iSafeRM/deploy/software/mysql/ansible && $(CONDA_ACTIVATE) activate mytunning &&  \
		ansible-playbook mysql_deploy.yml

.PHONY: clean-playbook-mysql
clean-playbook-mysql:
	@cd ~/iSafeRM/deploy/software/mysql/ansible && $(CONDA_ACTIVATE) activate mytunning && \
		ansible-playbook mysql_clean.yml


.PHONY: run-mysql
run-mysql:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/helper.py \
		-tn mysql_both_all_bo_tpe_2000 -b mysql -it 2000 -re 1 -para both -dr all -acq ei -algo bo_tpe


.PHONY: run-exp-sn
run-exp-sn:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python experiments/resource_config/tuning_up.py



.PHONY: run-exp-safescaler
run-exp-safescaler:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python experiments/SafeScaler/policy_test.py


.PHONY: run-rl-mm
run-rl-mm:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safetuning.py -tn mm

.PHONY: run-rl-hr
run-rl-hr:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safetuning.py -tn hr

.PHONY: run-rl-tt
run-rl-tt:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safetuning.py -tn tt

.PHONY: run-rl-sn
run-rl-sn:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python tunning/safetuning.py -tn sn


.PHONY: run-exp-find-key-services-sn
run-exp-find-key-services-sn:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python experiments/SafeScaler/find_key_services.py -tn sn

.PHONY: run-exp-find-key-services-mm
run-exp-find-key-services-mm:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python experiments/SafeScaler/find_key_services.py -tn mm

.PHONY: run-exp-find-key-services-hr
run-exp-find-key-services-hr:
	@cd ~/iSafeRM && $(CONDA_ACTIVATE) activate mytunning && \
		PYTHONPATH=. python experiments/SafeScaler/find_key_services.py -tn hr