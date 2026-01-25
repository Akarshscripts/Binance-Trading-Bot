"""
messenger.main
----------------
This module defines the main class which is used to create the messenger.

VERSION = "0.1.1"
LAST_UPDATED = "2026-01-25"
AUTHOR = "Prithvi Srivastava"
LICENSE = "MIT"

Classes
-------

Messenger:
    The main class which is used to create the messenger.
"""

# 1st party imports
import inspect
import logging
from typing import Dict, Any, Optional, List

# 3rd party imports
import requests

# local imports
from .exception_handler import stop_all_exceptions

# create a logger
LOGGER = logging.getLogger("Messenger")


class Messenger:
    """
    The main class which is used to create the messenger.

    Functions:
        __init__
            Initializes the messenger.

    """

    USERNAME = "Son of CHARLES"
    AVATAR_URL = (
        "https://i.pinimg.com/736x/c8/06/d5/c806d58879926807151297a162228544.jpg"
    )

    def __init__(self, webhook_url: str):
        """
        Initializes the messenger.

        Args:
            webhook_url (str): The discord webhook url.
        """

        # map the webhook url to the instance
        self.webhook_url = webhook_url

        # map the logger
        self.logger = LOGGER

    def __make_post_req(
        self, url: str, payload: Dict[str, Any], extra_headers: Dict[str, Any] = {}
    ) -> Optional[requests.Response]:
        """
        Makes a post request to the given url. Check the status code and return the json response.

        Args:
            url (str): The url to make the post request to.
            payload (Dict[str, Any]): The payload to send with the post request.
            extra_headers (Dict[str, Any], optional): The extra headers to send with the post request. Defaults to None.

        Returns:
            Optional[requests.Response]: The response object if the request was successful, None otherwise.
        """

        # create the headers
        headers = {"Content-Type": "application/json"}

        # add the extra headers
        if extra_headers:
            headers.update(extra_headers)

        # get the response
        response = requests.post(url, json=payload, headers=headers)

        # check for status and log error if any
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            self.logger.exception(
                f"Error while making a post request.",
                extra={
                    "event": "function",
                    "function": f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
                    "url": url,
                    "payload": payload,
                    "extra_headers": extra_headers,
                },
            )
            return None

        # return the response object
        return response

    def __get_chunks_from_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """
        Splits the text into chunks of 2000 characters with respect of last space.

        Args:
            text (str): The text to split.
            chunk_size (int, optional): The maximum size of the chunk. Defaults to 2000.

        Returns:
            List[str]: The chunks of text.
        """

        # store the chunks and the index of last space
        chunks = []
        last_split_idx = 0

        # split the message into chunks of 2000 characters with respect of last space
        while last_split_idx < len(text):

            # get the current text
            current_text_block = text[last_split_idx : last_split_idx + chunk_size]

            # if no more text is available
            if (last_split_idx + chunk_size) >= len(text):
                chunks.append(current_text_block)
                break

            # get the index of last space
            recent_space_idx = current_text_block.rfind(" ")
            if recent_space_idx == -1:
                self.logger.error(
                    "Failed to split the text into chunks. No spaces found in the text.",
                    extra={
                        "event": "function",
                        "function": f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
                        "text": text,
                        "chunk_size": chunk_size,
                    },
                )
                return None

            # create a chunk
            chunks.append(current_text_block[:recent_space_idx])

            # update the last split index
            last_split_idx += recent_space_idx + 1

        # return the chunks
        return chunks

    def __add_key_if_available(
        self, key: str, value: Any, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adds the key to the payload if the value is not None.

        Args:
            key (str): The key to add.
            value (Any): The value to add.
            payload (Dict[str, Any]): The payload to add the key to.
        """

        # add the key if the value is not None
        if value:
            payload[key] = value

        return payload

    def __get_image_embed_component(
        self,
        image_url: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        footer: Optional[str] = None,
        color: Optional[str] = None,
        embed_image_link: bool = False,
    ):
        """
        Creates an image embed component for a Discord message.

        Args:
            image_url (str): The URL of the image to embed.
            title (Optional[str]): The title of the embed, if any.
            description (Optional[str]): The description of the embed, if any.
            footer (Optional[str]): The footer text of the embed, if any.
            color (Optional[str]): The color code of the embed, if any.
            embed_image_link (bool): Whether to embed the image link. Defaults to False.

        Returns:
            dict: The payload dictionary representing the image embed component.
        """

        # create a base payload
        payload = {"image": {"url": image_url}}

        # add title, description and type
        payload = self.__add_key_if_available("title", title, payload)
        payload = self.__add_key_if_available("description", description, payload)
        payload = self.__add_key_if_available("type", "image", payload)

        # add footer if available
        if footer:
            payload["footer"] = {"text": footer}

        # add embed image link if available
        if embed_image_link:
            payload["url"] = image_url

        # add color if available
        if color:
            payload["color"] = color

        # return the payload
        return payload

    @stop_all_exceptions
    def send_text_message(
        self,
        message: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        max_chunk_size: int = 2000,
    ) -> bool:
        """
        Send a text message to the webhook.

        Args:
            message (str): The message to send.
            max_chunk_size (int, optional): The maximum size of the chunk. Defaults to 2000.
            username (Optional[str], optional): The username to send the message as. Defaults to None.
            avatar_url (Optional[str], optional): The avatar url to send the message as. Defaults to None.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """

        # check if the message is empty
        if len(message) == 0:
            self.logger.error(
                "Trying to send an empty message.",
                extra={
                    "event": "function",
                    "function": f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
                },
            )
            return True

        # get the chunks
        refined_message = message.lstrip()
        chunks = self.__get_chunks_from_text(refined_message, max_chunk_size)

        # update username and avatar url
        username = username or self.USERNAME
        avatar_url = avatar_url or self.AVATAR_URL

        # send the message
        for chunk in chunks:

            # create the payload
            payload = {"content": chunk, "username": username, "avatar_url": avatar_url}

            # make the post request
            response = self.__make_post_req(self.webhook_url, payload)

            # return the response
            if not isinstance(response, requests.Response):
                return False

        # return True
        return True

    @stop_all_exceptions
    def send_image_message(
        self,
        image_url: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        footer: Optional[str] = None,
        color: Optional[str] = None,
        embed_image_link: bool = False,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> bool:
        """
        Send an image message to the webhook.

        Args:
            image_url (str): The image url to send.
            username (Optional[str], optional): The username to send the message as. Defaults to None.
            avatar_url (Optional[str], optional): The avatar url to send the message as. Defaults to None.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """

        # trim the message title to 256 characters
        if title and len(title) > 256:
            title = title[:256]

        # trim the message description to 4096 characters
        if description and len(description) > 4096:
            description = description[:4096]

        # validate color
        if color and not color.isdigit():
            self.logger.error(
                "Invalid color code. Removing color.",
                extra={
                    "event": "function",
                    "function": f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}",
                    "color": color,
                },
            )
            color = None

        # create the payload
        payload = {
            "embeds": [
                self.__get_image_embed_component(
                    image_url=image_url,
                    title=title,
                    description=description,
                    footer=footer,
                    color=color,
                    embed_image_link=embed_image_link,
                )
            ],
            "username": username or self.USERNAME,
            "avatar_url": avatar_url or self.AVATAR_URL,
        }

        # make the post request
        response = self.__make_post_req(self.webhook_url, payload)

        # return the response
        if not isinstance(response, requests.Response):
            return False

        # return True
        return True
