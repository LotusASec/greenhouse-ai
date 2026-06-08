FROM grafana/grafana:10.4.0

USER root

# Install SQLite datasource plugin and fix permissions in one layer
RUN grafana-cli plugins install frser-sqlite-datasource && \
    chown -R grafana:root /var/lib/grafana/plugins && \
    chmod -R g+rX /var/lib/grafana/plugins

# Copy provisioning config and dashboards
COPY grafana/provisioning/ /etc/grafana/provisioning/
COPY grafana/dashboards/   /var/lib/grafana/dashboards/

RUN chown -R grafana:root /var/lib/grafana/dashboards /etc/grafana/provisioning

USER grafana

ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=greenhouse
ENV GF_AUTH_ANONYMOUS_ENABLED=true
ENV GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
ENV GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/edge_node.json
