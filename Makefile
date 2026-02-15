all: fe-env

fe-env:
	@python3 -m venv .fe-env/
	@.fe-env/bin/pip3 install -r requirements.txt

	@clear

	@echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-"
	@echo "Welcome to the GeoFin Framework!"
	@echo "Here you can optimize your portfolio or price financial derivatives,"
	@echo "but we recommend customizing your configuration file before getting "
	@echo "started. We provided one for guidance at `configs/config.yaml`."
	@echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-"

clean:
	@rm -rf .fe-env/ *data/ *exports/