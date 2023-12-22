import datetime
import sqlite3
import threading


class NoteTaker:
    db_path: str

    def __init__(self, db_path: str = "notes.db", debounce_time=10):
        """
        Create a new NoteTaker instance.

        :param db_path: The path to the database file.
        :param debounce_time: The time in seconds to wait before flushing the buffer to the database.
        """
        self.db_path = db_path
        self.connection = self.create_connection()
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS notes (text TEXT, created_at TEXT)"
        )

        self.buffer = []
        self.debounce_time = debounce_time
        self.timer = None

    def create_connection(self):
        return sqlite3.connect(self.db_path)

    def append(self, text: str) -> None:
        """
        Append text to the note, save it to the database along with the creation datetime.

        :param text: The text to append to the note.
        """
        now = datetime.datetime.now().isoformat()
        self.buffer.append((text, now))
        if self.timer is None:
            self.timer = threading.Timer(self.debounce_time, self.flush)
            self.timer.start()

    def flush(self):
        print("Flushing buffer to database...")
        connection = self.create_connection()
        cursor = connection.cursor()
        cursor.executemany("INSERT INTO notes VALUES (?, ?)", self.buffer)
        connection.commit()
        self.buffer = []
        self.timer = None
        cursor.close()
        connection.close()

    def __del__(self):
        if self.timer is not None:
            self.timer.cancel()
        self.flush()
        self.cursor.close()
        self.connection.close()
