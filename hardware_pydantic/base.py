from __future__ import annotations

from typing import Any, Literal, Type
# only used for drawing instruction DAG
from N2G import drawio_diagram
from pydantic import BaseModel, Field

from .utils import str_uuid

DEVICE_ACTION_METHOD_PREFIX = "action__"
DEVICE_ACTION_METHOD_ACTOR_TYPE = Literal['pre', 'post', 'proj']

"""Base classes for hardware units."""


class Individual(BaseModel):
    """A thing with an identifier."""

    identifier: str = Field(default_factory=str_uuid)

    def __hash__(self):
        """Hash by identifier."""
        return hash(self.identifier)

    def __eq__(self, other: Individual):
        """Compare by identifier."""
        return self.identifier == other.identifier


class LabObject(Individual):
    """
    A lab object is a physical object in a lab that
    1. is not a chemical designed to participate in reactions
    2. has constant, physically defined 3D boundary
    3. has a (non empty) set of measurable qualities that need to be tracked, this set is the state of this `LabObject`
        # TODO can we have a pydantic model history tracker? similar to https://pypi.org/project/pydantic-changedetect/
        # TODO mutable fields vs immutable fields?
    """

    @property
    def state(self) -> dict:
        """The current state of this lab object as a dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("layout"):
                continue
            d[k] = v
        return d

    def validate_state(self, state: dict) -> bool:
        """Validate a state dictionary."""
        pass

    def validate_current_state(self) -> bool:
        """Validate the current state of this lab object."""
        return self.validate_state(self.state)


class PreActError(Exception):
    """Error raised when pre-actor method fails."""
    pass


class PostActError(Exception):
    """Error raised when post-actor method fails."""
    pass


class Device(LabObject):
    """
    A `LabObject` is a physical object in a lab that
    1. can receive instructions
    2. can change its state and other lab objects' states using its action methods,
    3. cannot change another device's state # TODO does this actually matter?
    """

    @property
    def action_names(self) -> list[str]:
        """A sorted list of the names of all defined actions."""
        names = sorted(
            {k[len(DEVICE_ACTION_METHOD_PREFIX):] for k in dir(self) if k.startswith(DEVICE_ACTION_METHOD_PREFIX)}
        )
        return names

    def action__dummy(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            **kwargs
    ) -> tuple[list[LabObject], float] | None:
        """
        A dummy action method that does nothing,
        but can be used as a template for other action methods.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The type of actor method to call, one of 'pre', 'post', 'proj'. Default is 'pre'.
        **kwargs : dict
            The parameters for the action method.
        """
        assert 'actor_type' not in kwargs
        if actor_type == 'pre':
            # the pre actor method of an action should
            # - check the current states of involved objects, raise PreActorError if not met
            return
        elif actor_type == 'post':
            # the post actor method of an action should
            # - make state transitions for involved objects, raise PostActorError if illegal transition
            return
        elif actor_type == 'proj':
            # the projection method of an action should
            # - return a list of all involved objects, except self
            # - return the projected time cost
            return [], 0
        else:
            raise ValueError

    def act(
            self,
            action_name: str = "dummy",
            actor_type: Literal['pre', 'post', 'proj'] = 'pre',
            action_parameters: dict[str, Any] = None,
    ):
        """
        Get the action method of this device by name and call it.

        Parameters
        ----------
        action_name : str
            The name of the action method to call. Default is 'dummy'.
        actor_type : Literal['pre', 'post', 'proj']
            The type of actor method to call, one of 'pre', 'post', 'proj'. Default is 'pre'.
        action_parameters : dict[str, Any]
            The parameters for the action method. Default is None.
        """
        assert action_name in self.action_names, f"{action_name} not in {self.action_names}"
        if action_parameters is None:
            action_parameters = dict()
        method_name = DEVICE_ACTION_METHOD_PREFIX + action_name

        return getattr(self, method_name)(actor_type=actor_type, **action_parameters)

    def act_by_instruction(self, instruct: Instruction, actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE):
        """Perform action with an instruction.

        Parameters
        ----------
        instruct : Instruction
            The instruction to perform.
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The type of actor method to call, one of 'pre', 'post', 'proj'.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The result of the action method.
        """
        assert instruct.device == self
        return self.act(action_name=instruct.action_name, action_parameters=instruct.action_parameters, actor_type=actor_type)


class Instruction(Individual):
    """
    This class defines an instruction sent to a device for an action.

    Parameters
    ----------
    device : Device
        The device to send the instruction to.
    action_parameters : dict
        The parameters for the action method of the device.
    action_name : str
        The name of the action method of the device.
    description : str
        The description of this instruction.

    Note
    ----

    instruction:
    - an instruction is sent to and received by one and only one `Device` instance (the `actor`) instantly
    - an instruction requests one and only one `action_name` from the `Device` instance
    - an instruction contains static parameters that the `action_method` needs
    - an instruction can involve zero or more `LabObject` instances
    - an instruction cannot involve any `Device` instance except the `actor`

    action:
    - an action is a physical process performed following an instruction
    - an action
        - starts when
            - the actor is available, and
            - the action is at the top of the queue of that actor
        - ends when
            - the duration, returned by the action method of the actor, has passed
    """
    device: Device
    action_parameters: dict = dict()
    action_name: str = "dummy"
    description: str = ""

    preceding_type: Literal["ALL", "ANY"] = "ALL"
    # TODO this has no effect as it is not passed to casymda

    preceding_instructions: list[str] = []


class Lab(BaseModel):
    """
    A `Lab` is a collection of `LabObject` instances and operations on them.

    Parameters
    ----------
    dict_object : dict[str, LabObject | Device]
        A dictionary of all `LabObject` instances in this `Lab`.
    dict_instruction : dict[str, Instruction]
        A dictionary of all `Instruction` instances in this `Lab`.
    """

    dict_instruction: dict[str, Instruction] = dict()
    dict_object: dict[str, LabObject | Device] = dict()

    def __getitem__(self, identifier: str):
        """Getting item by key from dict_object."""
        return self.dict_object[identifier]

    def __setitem__(self, key, value):
        """Setting an item with key and value pair to dict_object."""
        raise NotImplementedError

    def act_by_instruction(self,
                           instruct: Instruction,
                           actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE):
        """Perform action with an instruction.

        Parameters
        ----------
        instruct : Instruction
            The instruction to perform.
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The type of actor method to call, one of 'pre', 'post', 'proj'.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The result of the action method.
        """
        # make sure we are working on the same device
        actor = self.dict_object[instruct.device.identifier]
        assert isinstance(actor, Device)
        return actor.act_by_instruction(instruct, actor_type=actor_type)

    def add_instruction(self, instruct: Instruction):
        """Add an instruction to the lab.

        Parameters
        ----------
        instruct : Instruction
            The instruction to add.
        """
        assert instruct.identifier not in self.dict_instruction
        self.dict_instruction[instruct.identifier] = instruct

    def remove_instruction(self, instruct: Instruction | str):
        """Remove an instruction from the lab.

        Parameters
        ----------
        instruct : Instruction | str
            The instruction to remove.
        """
        if isinstance(instruct, str):
            assert instruct in self.dict_instruction
            self.dict_instruction.pop(instruct)
        else:
            assert instruct.identifier in self.dict_instruction
            self.dict_instruction.pop(instruct.identifier)

    def add_object(self, d: LabObject | Device):
        """Add a `LabObject` or `Device` instance to the lab.

        Parameters
        ----------
        d : LabObject | Device
            The `LabObject` or `Device` instance to add.
        """
        assert d.identifier not in self.dict_object
        self.dict_object[d.identifier] = d

    def remove_object(self, d: LabObject | Device | str):
        """Remove a `LabObject` or `Device` instance from the lab.

        Parameters
        ----------
        d : LabObject | Device | str
            The `LabObject` or `Device` instance to remove.
        """
        if isinstance(d, str):
            assert d in self.dict_object
            self.dict_object.pop(d)
        else:
            assert d.identifier in self.dict_object
            self.dict_object.pop(d.identifier)

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        """Get the state of the lab.

        Returns
        -------
        dict[str, dict[str, Any]]
            The state of the lab as a dictionary where the keys are the identifiers of the
            `LabObject` instances and the values are the states of the `LabObject` instances.
        """
        return {d.identifier: d.state for d in self.dict_object.values()}

    def dict_object_by_class(self, object_class: Type):
        """Get a dictionary of `LabObject` instances of a certain class.

        Parameters
        ----------
        object_class : Type
            The class of the `LabObject` instances to get.

        Returns
        -------
        dict[str, LabObject]
            A dictionary of `LabObject` instances of the specified class.
        """
        return {k: v for k, v in self.dict_object.items() if v.__class__ == object_class}

    def __repr__(self):
        """Get a string representation of the lab, including the `LabObject` and `Device` and
        the corresponding states."""
        return "\n".join([f"{obj.identifier}: {obj.state}" for obj in self.dict_object.values()])

    def __str__(self):
        """Get a string representation of the lab, including the `LabObject` and `Device` and the
        corresponding states."""
        return self.__repr__()

    @property
    def instruction_graph(self) -> drawio_diagram:
        """Get a diagram of the instruction graph.

        Returns
        -------
        drawio_diagram
            The instruction graph.

        """
        diagram = drawio_diagram()
        diagram.add_diagram("Page-1")

        for k, ins in self.dict_instruction.items():
            diagram.add_node(id=f"{ins.identifier}\n{ins.description}")

        for ins in self.dict_instruction.values():
            for dep in ins.preceding_instructions:
                pre_ins = self.dict_instruction[dep]
                this_ins_node = f"{ins.identifier}\n{ins.description}"
                pre_ins_node = f"{pre_ins.identifier}\n{pre_ins.description}"
                diagram.add_link(pre_ins_node, this_ins_node, style="endArrow=classic")
        return diagram
