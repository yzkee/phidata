from typing import Optional, Any, Dict, List, Union

from phidata.aws.api_client import AwsApiClient
from phidata.aws.resource.base import AwsResource, AwsObject
from phidata.aws.resource.ec2.subnet import Subnet
from phidata.resource.reference import AwsReference
from phidata.utils.cli_console import print_info, print_error, print_warning
from phidata.utils.log import logger


class InboundRule(AwsObject):
    """
    Reference:
        - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/authorize_security_group_ingress.html
    """

    # What to enable ingress for.
    # The IPv4 CIDR range. You can either specify a CIDR range or a source security group, not both.
    # To specify a single IPv4 address, use the /32 prefix length.
    cidr_ip: Optional[str] = None
    # The IPv6 CIDR range. You can either specify a CIDR range or a source security group, not both.
    # To specify a single IPv6 address, use the /128 prefix length.
    cidr_ipv6: Optional[str] = None
    # The security group id to allow access from.
    source_security_group_id: Optional[Union[str, AwsReference]] = None
    # The security group name to allow access from.
    # For a security group in a nondefault VPC, use the security group ID.
    source_security_group_name: Optional[str] = None
    # A description for this security group rule
    description: Optional[str] = None

    # The port to allow access from.
    # If provided, sets both from_port and to_port.
    port: Optional[int] = None
    # The port range to allow access from.
    from_port: Optional[int] = None
    # The port range to allow access from.
    to_port: Optional[int] = None
    # The protocol to allow access from. Default is tcp.
    ip_protocol: Optional[str] = None


class OutboundRule(AwsObject):
    """
    Reference:
        - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/authorize_security_group_ingress.html
    """

    # What to enable egress for.
    # The IPv4 CIDR range. You can either specify a CIDR range or a source security group, not both.
    # To specify a single IPv4 address, use the /32 prefix length.
    cidr_ip: Optional[str] = None
    # The IPv6 CIDR range. You can either specify a CIDR range or a source security group, not both.
    # To specify a single IPv6 address, use the /128 prefix length.
    cidr_ipv6: Optional[str] = None
    # The security group id to allow access from.
    source_security_group_id: Optional[Union[str, AwsReference]] = None
    # The security group name to allow access from.
    # For a security group in a nondefault VPC, use the security group ID.
    source_security_group_name: Optional[str] = None
    # A description for this security group rule
    description: Optional[str] = None

    # The port to allow access from.
    # If provided, sets both from_port and to_port.
    port: Optional[int] = None
    # The port range to allow access from.
    from_port: Optional[int] = None
    # The port range to allow access from.
    to_port: Optional[int] = None
    # The protocol to allow access from. Default is tcp.
    ip_protocol: Optional[str] = None


class SecurityGroup(AwsResource):
    """
    Reference:
        - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/securitygroup/index.html
        - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2/client/create_security_group.html
    """

    resource_type = "SecurityGroup"
    service_name = "ec2"

    # The name of the security group.
    name: str
    # A description for the security group.
    description: Optional[str] = None
    # The ID of the VPC for the security group.
    vpc_id: Optional[str] = None
    # Derive the vpc_id from the subnets.
    # When more than one subnet is provided, both must be in the same VPC.
    subnets: Optional[List[Union[str, Subnet]]] = None
    # The tags to assign to the security group.
    tag_specifications: Optional[list] = None
    # Checks whether you have the required permissions for the action,
    # without actually making the request, and provides an error response.
    # If you have the required permissions, the error response is DryRunOperation.
    # Otherwise, it is UnauthorizedOperation.
    dry_run: Optional[bool] = None

    # The inbound rules associated with the security group.
    inbound_rules: Optional[List[InboundRule]] = None
    # The IP permissions to authorize ingress for
    ingress_ip_permissions: Optional[List[Dict[str, Any]]] = None
    # The outbound rules associated with the security group.
    outbound_rules: Optional[List[OutboundRule]] = None
    # The IP permissions to authorize egress for
    egress_ip_permissions: Optional[List[Dict[str, Any]]] = None

    # Security Group id
    group_id: Optional[str] = None

    def _create(self, aws_client: AwsApiClient) -> bool:
        """Creates the SecurityGroup

        Args:
            aws_client: The AwsApiClient for the current Security group
        """
        print_info(f"Creating {self.get_resource_type()}: {self.get_resource_name()}")

        # Step 1: Build Security group configuration
        # create a dict of args which are not null, otherwise aws type validation fails
        not_null_args: Dict[str, Any] = {}

        # Build description
        description = self.description or "Created by phi"
        if description is not None:
            not_null_args["Description"] = description

        # Get vpc_id
        vpc_id = self.vpc_id
        if vpc_id is None and self.subnets is not None:
            from phidata.aws.resource.ec2.subnet import get_vpc_id_from_subnet_ids

            subnet_ids = []
            for subnet in self.subnets:
                if isinstance(subnet, Subnet):
                    subnet_ids.append(subnet.id)
                elif isinstance(subnet, str):
                    subnet_ids.append(subnet)
            vpc_id = get_vpc_id_from_subnet_ids(subnet_ids, aws_client)
        if vpc_id is not None:
            not_null_args["VpcId"] = vpc_id

        if self.tag_specifications:
            not_null_args["TagSpecifications"] = self.tag_specifications
        if self.dry_run:
            not_null_args["DryRun"] = self.dry_run

        # Step 2: Create Security group
        service_client = self.get_service_client(aws_client)
        try:
            create_response = service_client.create_security_group(
                GroupName=self.name,
                **not_null_args,
            )
            logger.debug(f"Response: {create_response}")

            # Validate resource creation
            if create_response is not None:
                print_info(f"SecurityGroup created: {self.get_resource_name()}")
                self.active_resource = create_response
                return True
        except Exception as e:
            print_error(f"{self.get_resource_type()} could not be created.")
            print_error(e)
        return False

    def post_create(self, aws_client: AwsApiClient) -> bool:
        # Wait for SecurityGroup to be created
        if self.wait_for_creation:
            try:
                print_info(f"Waiting for {self.get_resource_type()} to be created.")
                waiter = self.get_service_client(aws_client).get_waiter(
                    "security_group_exists"
                )
                waiter.wait(
                    Filters=[
                        {
                            "Name": "group-name",
                            "Values": [self.name],
                        },
                    ],
                    WaiterConfig={
                        "Delay": self.waiter_delay,
                        "MaxAttempts": self.waiter_max_attempts,
                    },
                )
            except Exception as e:
                print_error("Waiter failed.")
                print_error(e)
                return False

        # Add inbound rules
        if self.inbound_rules is not None or self.ingress_ip_permissions:
            _success = self.add_inbound_rules(aws_client)
            if not _success:
                return False
        # Add outbound rules
        if self.outbound_rules is not None or self.egress_ip_permissions:
            _success = self.add_outbound_rules(aws_client)
            if not _success:
                return False
        return True

    def _read(self, aws_client: AwsApiClient) -> Optional[Any]:
        """Reads the SecurityGroup

        Args:
            aws_client: The AwsApiClient for the current session
        """
        from botocore.exceptions import ClientError

        logger.debug(f"Reading {self.get_resource_type()}: {self.get_resource_name()}")
        service_client = self.get_service_client(aws_client)
        try:
            describe_response = service_client.describe_security_groups(
                Filters=[
                    {
                        "Name": "group-name",
                        "Values": [self.name],
                    },
                ],
            )
            logger.debug(f"Response: {describe_response}")
            resource_list = describe_response.get("SecurityGroups", None)

            if resource_list is not None and isinstance(resource_list, list):
                for resource in resource_list:
                    if resource.get("GroupName", None) == self.name:
                        self.active_resource = resource
        except ClientError as ce:
            logger.debug(f"ClientError: {ce}")
        except Exception as e:
            print_error(f"Error reading {self.get_resource_type()}.")
            print_error(e)
        return self.active_resource

    def _delete(self, aws_client: AwsApiClient) -> bool:
        """Deletes the SecurityGroup

        Args:
            aws_client: The AwsApiClient for the current session
        """
        from botocore.exceptions import ClientError

        print_info(f"Deleting {self.get_resource_type()}: {self.get_resource_name()}")

        service_client = self.get_service_client(aws_client)
        self.active_resource = None

        try:
            group_id = self.get_security_group_id(aws_client)
            if group_id is not None:
                delete_response = service_client.delete_security_group(GroupId=group_id)
            else:
                delete_response = service_client.delete_security_group(
                    GroupName=self.name
                )
            logger.debug(f"Response: {delete_response}")

            print_info(
                f"{self.get_resource_type()}: {self.get_resource_name()} deleted"
            )
            return True
        except ClientError as ce:
            ce_resp = ce.response
            if ce_resp is not None:
                if ce_resp.get("Error", {}).get("Code", "") == "DependencyViolation":
                    logger.warning(
                        f"SecurityGroup {self.get_resource_name()} could not be deleted as it is being used by another resource."
                    )
                    logger.warning(
                        "Please try again later or delete resources manually."
                    )
                    logger.warning(f"Error: {ce_resp}")
                    return True
        except Exception as e:
            print_error(f"{self.get_resource_type()} could not be deleted.")
            print_error("Please try again or delete resources manually.")
            print_error(e)
        return False

    def get_security_group_id(
        self, aws_client: Optional[AwsApiClient] = None
    ) -> Optional[str]:
        """Returns the security group id"""

        if self.group_id is not None:
            return self.group_id

        resource = self.read(aws_client)
        if resource is not None:
            self.group_id = resource.get("GroupId", None)

        return self.group_id

    def add_inbound_rules(self, aws_client: AwsApiClient) -> bool:
        """Adds the specified inbound (ingress) rules to a security group.

        Args:
            aws_client: The AwsApiClient for the current session
        """
        from botocore.exceptions import ClientError

        # create a dict of args which are not null, otherwise aws type validation fails
        not_null_args: Dict[str, Any] = {}

        group_id = self.get_security_group_id(aws_client)
        if group_id is None:
            logger.warning(f"GroupId for {self.get_resource_name()} not found.")
            return False
        not_null_args["GroupId"] = group_id

        ip_permissions: List[Dict[str, Any]] = self.ingress_ip_permissions or []
        if self.inbound_rules is not None:
            for rule in self.inbound_rules:
                ip_permission: Dict[str, Any] = {
                    "IpProtocol": rule.ip_protocol or "tcp"
                }
                if rule.from_port is not None:
                    ip_permission["FromPort"] = rule.from_port
                if rule.to_port is not None:
                    ip_permission["ToPort"] = rule.to_port
                if rule.port is not None:
                    ip_permission["FromPort"] = rule.port
                    ip_permission["ToPort"] = rule.port
                if rule.cidr_ip is not None:
                    ip_permission["IpRanges"] = [
                        {
                            "CidrIp": rule.cidr_ip,
                            "Description": rule.description or "",
                        },
                    ]
                if rule.cidr_ipv6 is not None:
                    ip_permission["Ipv6Ranges"] = [
                        {
                            "CidrIpv6": rule.cidr_ipv6,
                            "Description": rule.description or "",
                        },
                    ]
                if (
                    rule.source_security_group_id is not None
                    or rule.source_security_group_name is not None
                ):
                    source_sg_id: Optional[str] = None
                    if isinstance(rule.source_security_group_id, str):
                        source_sg_id = rule.source_security_group_id
                    elif isinstance(rule.source_security_group_id, AwsReference):
                        source_sg_id = rule.source_security_group_id.get_reference(
                            aws_client=aws_client
                        )

                    user_id_group_pair = {}
                    if source_sg_id is not None:
                        user_id_group_pair["GroupId"] = source_sg_id
                    if rule.source_security_group_name is not None:
                        user_id_group_pair[
                            "GroupName"
                        ] = rule.source_security_group_name
                    if rule.description is not None:
                        user_id_group_pair["Description"] = rule.description
                    ip_permission["UserIdGroupPairs"] = [user_id_group_pair]
                logger.debug(f"Inbound Rule: {ip_permission}")
                ip_permissions.append(ip_permission)

        if len(ip_permissions) == 0:
            logger.debug(f"No ingress rules found for {self.get_resource_name()}")
            return True
        if ip_permissions is not None:
            not_null_args["IpPermissions"] = ip_permissions

        if self.dry_run is not None:
            not_null_args["DryRun"] = True

        service_client = self.get_service_client(aws_client)
        try:
            response = service_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=ip_permissions,
            )
            logger.debug(f"Response: {response}")

            # Validate the response
            if response is not None and response.get("Return", False):
                print_info(f"Ingress rules added to {self.get_resource_name()}")
                return True
        except ClientError as ce:
            ce_resp = ce.response
            if ce_resp is not None:
                if (
                    ce_resp.get("Error", {}).get("Code", "")
                    == "InvalidPermission.Duplicate"
                ):
                    print_info(
                        f"Ingress rules already exist for {self.get_resource_name()}"
                    )
                    return True
            logger.debug(f"ClientError: {ce}")
        except Exception as e:
            logger.warning(
                f"Ingress rules could not be added to {self.get_resource_name()}: {e}"
            )
        return False

    def add_outbound_rules(self, aws_client: AwsApiClient) -> bool:
        """Adds the specified outbound (egress) rules to a security group.

        Args:
            aws_client: The AwsApiClient for the current session
        """
        from botocore.exceptions import ClientError

        # create a dict of args which are not null, otherwise aws type validation fails
        not_null_args: Dict[str, Any] = {}

        group_id = self.get_security_group_id(aws_client)
        if group_id is None:
            logger.warning(f"GroupId for {self.get_resource_name()} not found.")
            return False
        not_null_args["GroupId"] = group_id

        ip_permissions: List[Dict[str, Any]] = self.egress_ip_permissions or []
        if self.outbound_rules is not None:
            for rule in self.outbound_rules:
                ip_permission: Dict[str, Any] = {
                    "IpProtocol": rule.ip_protocol or "tcp"
                }
                if rule.from_port is not None:
                    ip_permission["FromPort"] = rule.from_port
                if rule.to_port is not None:
                    ip_permission["ToPort"] = rule.to_port
                if rule.port is not None:
                    ip_permission["FromPort"] = rule.port
                    ip_permission["ToPort"] = rule.port
                if rule.cidr_ip is not None:
                    ip_permission["IpRanges"] = [
                        {
                            "CidrIp": rule.cidr_ip,
                            "Description": rule.description or "",
                        },
                    ]
                if rule.cidr_ipv6 is not None:
                    ip_permission["Ipv6Ranges"] = [
                        {
                            "CidrIpv6": rule.cidr_ipv6,
                            "Description": rule.description or "",
                        },
                    ]
                if (
                    rule.source_security_group_id is not None
                    or rule.source_security_group_name is not None
                ):
                    source_sg_id: Optional[str] = None
                    if isinstance(rule.source_security_group_id, str):
                        source_sg_id = rule.source_security_group_id
                    elif isinstance(rule.source_security_group_id, AwsReference):
                        source_sg_id = rule.source_security_group_id.get_reference(
                            aws_client=aws_client
                        )

                    user_id_group_pair = {}
                    if source_sg_id is not None:
                        user_id_group_pair["GroupId"] = source_sg_id
                    if rule.source_security_group_name is not None:
                        user_id_group_pair[
                            "GroupName"
                        ] = rule.source_security_group_name
                    if rule.description is not None:
                        user_id_group_pair["Description"] = rule.description
                    ip_permission["UserIdGroupPairs"] = [user_id_group_pair]
                logger.debug(f"Outbound Rule: {ip_permission}")
                ip_permissions.append(ip_permission)

        if len(ip_permissions) == 0:
            logger.debug(f"No ingress rules found for {self.get_resource_name()}")
            return True
        if ip_permissions is not None:
            not_null_args["IpPermissions"] = ip_permissions

        if self.dry_run is not None:
            not_null_args["DryRun"] = True

        service_client = self.get_service_client(aws_client)
        try:
            response = service_client.authorize_security_group_egress(
                GroupId=group_id,
                IpPermissions=ip_permissions,
            )
            logger.debug(f"Response: {response}")

            # Validate the response
            if response is not None and response.get("Return", False):
                print_info(f"Egress rules added to {self.get_resource_name()}")
                return True
        except ClientError as ce:
            ce_resp = ce.response
            if ce_resp is not None:
                if (
                    ce_resp.get("Error", {}).get("Code", "")
                    == "InvalidPermission.Duplicate"
                ):
                    print_info(
                        f"Ingress rules already exist for {self.get_resource_name()}"
                    )
                    return True
            logger.debug(f"ClientError: {ce}")
        except Exception as e:
            logger.warning(
                f"Egress rules could not be added to {self.get_resource_name()}: {e}"
            )
        return False