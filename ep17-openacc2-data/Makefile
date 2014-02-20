# C compiler
CC = pgcc
OPT = -fast
MP  = -mp
OMP_THRDS = 6
ACC = -acc -Minfo=accel
TIME = /usr/bin/time --verbose
DIFF = diff --brief 

all: build run

build: hotspot_omp hotspot_acc

run: small medium large

small: small_omp small_acc
medium: medium_omp medium_acc
large: large_omp large_acc

hotspot_acc: hotspot.c
	$(CC) $(OPT) $(ACC) hotspot.c -o hotspot_acc

hotspot_omp: hotspot.c
	$(CC) $(OPT) $(MP) hotspot.c -o hotspot_omp

small_omp: hotspot_omp
	$(TIME) ./hotspot_omp  64 64 10000 $(OMP_THRDS) data/temp_64 data/power_64 > output_sm_omp.log
	$(DIFF) output_sm_omp.log data/sm_output.log

medium_omp: hotspot_omp
	$(TIME) ./hotspot_omp  512 512 10000 $(OMP_THRDS) data/temp_512 data/power_512 > output_md_omp.log
	$(DIFF) output_md_omp.log data/md_output.log

large_omp: hotspot_omp
	$(TIME) ./hotspot_omp  1024 1024 10000 $(OMP_THRDS) data/temp_1024 data/power_1024 > output_lg_omp.log
	$(DIFF) output_lg_omp.log data/lg_output.log

small_acc: hotspot_acc
	$(TIME) ./hotspot_acc 64 64 10000 1 data/temp_64 data/power_64 > output_sm_acc.log
	$(DIFF) output_sm_acc.log data/sm_output.log

medium_acc: hotspot_acc
	$(TIME) ./hotspot_acc  512 512 10000 1 data/temp_512 data/power_512 > output_md_acc.log
	$(DIFF) output_md_acc.log data/md_output.log

large_acc: hotspot_acc
	$(TIME) ./hotspot_acc  1024 1024 10000 1 data/temp_1024 data/power_1024 > output_lg_acc.log
	$(DIFF) output_lg_acc.log data/lg_output.log

clean:
	rm -f hotspot_acc hotspot_omp *.o *.log

