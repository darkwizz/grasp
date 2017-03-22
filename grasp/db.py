#---- CSV -----------------------------------------------------------------------------------------
# A comma-separated values file (CSV) stores table data as plain text.
# Each line in the file is a row in a table.
# Each row consists of column fields, separated by a comma.
import codecs
import collections
import inspect
import os

import csv as csvlib
import sqlite3 as sqlite

import re

from grasp.strings2 import decode_string


class matrix(list):

    def __getitem__(self, i):
        """ A 2D list with advanced slicing: matrix[row1:row2, col1:col2].
        """
        if isinstance(i, tuple):
            i, j = i
            if isinstance(i, slice):
                return [v[j] for v in list.__getitem__(self, i)]
            return list.__getitem__(self, i)[j]
        return list.__getitem__(self, i)

    @property
    def html(self):
        a = ['<table>']
        for r in self:
            a.append('<tr>')
            a.extend('<td>%s</td>' % v for v in r)
            a.append('</tr>')
        a.append('</table>')
        return u'\n'.join(a)

# m = matrix()
# m.append([1, 2, 3])
# m.append([4, 5, 6])
# m.append([7, 8, 9])
#
# print(m)        # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(m[0])     # [1, 2, 3]
# print(m[0,0])   #  1
# print(m[:,0])   # [1, 4, 7]
# print(m[:2,:2]) # [[1, 2], [4, 5]]


class CSV(matrix):

    def __init__(self, name='', separator=',', rows=[]):
        """ Returns the given .csv file as a list of rows, each a list of values.
        """
        try:
            self.name      = name
            self.separator = separator
            self._load()
        except IOError:
            pass # doesn't exist (yet)
        if rows:
            self.extend(rows)

    def _load(self):
        with open(self.name, 'r') as f:
            for r in csvlib.reader(f, delimiter=self.separator):
                r = [decode_string(v) for v in r]
                self.append(r)

    def save(self, name=''):
        a = []
        for r in self:
            r = ('"' + decode_string(s).replace('"', '""') + '"' for s in r)
            r = self.separator.join(r)
            a.append(r)
        f = codecs.open(name or self.name, 'w', encoding='utf-8')
        f.write('\n'.join(a))
        f.close()

    def update(self, rows=[], index=0):
        """ Appends the rows that have no duplicates in the given column(s).
        """
        u = set(map(repr, self[:,index])) # unique + hashable slices (slow)
        for r in rows:
            k = repr(r[index])
            if k not in u:
                self.append(r)
                u.add(k)

    def clear(self):
        list.__init__(self, [])

csv = CSV

# data = csv('test.csv')
# data.append([1, 'hello'])
# data.save()
#
# print(data[0,0]) # 1st cell
# print(data[:,0]) # 1st column


def col(i, a):
    """ Returns the i-th column in the given list of lists.
    """
    for r in a:
        yield r[i]


def cd(*args):
    """ Returns the directory of the script that calls cd() + given relative path.
    """
    f = inspect.currentframe()
    f = inspect.getouterframes(f)[1][1]
    f = f != '<stdin>' and f or os.getcwd()
    p = os.path.realpath(f)
    p = os.path.dirname(p)
    p = os.path.join(p, *args)
    return p

# print(cd('test.csv'))

#---- SQL -----------------------------------------------------------------------------------------
# A database is a collection of tables, with rows and columns of structured data.
# Rows can be edited or selected with SQL statements (Structured Query Language).
# Rows can be indexed for faster retrieval or related to other tables.

# SQLite is a lightweight engine for a portable database stored as a single file.

# https://www.sqlite.org/datatype3.html
AFFINITY = collections.defaultdict(
    lambda : 'text'    , {
       str : 'text'    ,
   unicode : 'text'    ,
     bytes : 'blob'    ,
      bool : 'integer' ,
       int : 'integer' ,
     float : 'real'
})


def schema(table, *fields, **type):
    """ Returns an SQL CREATE TABLE statement,
        with indices on fields ending with '*'.
    """
    s = 'create table if not exists `%s` (' % table + 'id integer primary key);'
    i = 'create index if not exists `%s_%s` on `%s` (`%s`);'
    for k in fields:
        k = re.sub(r'\*$', '', k)  # 'name*' => 'name'
        v = AFFINITY[type.get(k)]  #     str => 'text'
        s = s[:-2] + ', `%s` %s);' % (k, v)
    for k in fields:
        if k.endswith('*'):
            s += i % ((table, k[:-1]) * 2)
    return s

# print(schema('persons', 'name*', 'age', age=int))


class DatabaseError(Exception):
    pass


class Database(object):

    def __init__(self, name, schema=None, timeout=10, factory=sqlite.Row):
        """ SQLite database interface.
        """
        self.connection = sqlite.connect(name, timeout)
        self.connection.row_factory = factory
        if schema:
            for q in schema.split(";"):
                self(q, commit=False)
            self.commit()

    def __call__(self, sql, values=(), first=False, commit=True):
        """ Executes the given SQL statement.
        """
        try:
            r = self.connection.cursor().execute(sql, values)
            if commit:
                self.connection.commit()
            if first:
                return r.fetchone() if r else r  # single row
            else:
                return r
        except Exception as e:
            raise DatabaseError(str(e))

    def execute(self, *args, **kwargs):
        return self(*args, **kwargs)

    def commit(self):
        return self.connection.commit()

    def rollback(self):
        return self.connection.rollback()

    @property
    def id(self):
        return self('select last_insert_rowid()').fetchone()[0]

    def find(self, table, *fields, **filters):
        return self(*SELECT(table, *fields, **filters))

    def first(self, table, *fields, **filters):
        return self(*SELECT(table, *fields, **filters), first=True)

    def append(self, table, **fields):
        return self(*INSERT(table, **fields),
                        commit=fields.pop('commit', True))

    def update(self, table, id, **fields):
        return self(*UPDATE(table, id, **fields),
                        commit=fields.pop('commit', True))

    def remove(self, table, id, **fields):
        return self(*DELETE(table, id),
                        commit=fields.pop('commit', True))

    def __del__(self):
        try:
            self.connection.commit()
            self.connection.close()
            self.connection = None
        except:
            pass

# db = Database(cd('test.db'), schema('persons', 'name*', 'age', age=int))
# db.append('persons', name='Tom', age=30)
# db.append('persons', name='Guy', age=30)
#
# for id, name, age in db.find('persons', age='>20'):
#     print(name, age)


def concat(a, format='%s', separator=', '):
  # concat([1, 2, 3]) => '1, 2, 3'
    return separator.join(format % v for v in a)


def SELECT(table, *fields, **where):
    """ Returns an SQL SELECT statement + parameters.
    """

    def op(v):
        if isinstance(v, basestring) and re.search(r'^<=|>=', v): # '<10'
            return v[:2], v[2:]
        if isinstance(v, basestring) and re.search(r'^<|>', v): # '<10'
            return v[:1], v[1:]
        if isinstance(v, basestring) and re.search(r'\*', v): # '*ly'
            return 'like', v.replace('*', '%')
        if hasattr(v, '__iter__'):
            return 'in', v
        else:
            return '=', v

    s = 'select %s from %s where %s ' + 'limit %i, %i order by `%s`;' % (
         where.pop('slice', (0, -1)) + (
         where.pop('sort', 'id'),)
    )
    f = concat(fields or '*')
    k = where.keys()    # ['name', 'age']
    v = where.values()  # ['Tom*', '>10']
    v = map(op, v)      # [('like', 'Tom%'), ('>', '10')]
    v = zip(*v)         #  ('like', '>'), ('Tom%', '10')
    v = iter(v)
    q = next(v, ())
    v = next(v, ())
    s = s % (f, table, concat(zip(k, q), '`%s` %s ?', 'and'))
    s = s.replace('limit 0, -1 ', '', 1)
    s = s.replace('where  ', '', 1)
    return s, tuple(v)

# print(SELECT('persons', '*', age='>10', slice=(0, 10)))

def INSERT(table, **fields):
    """ Returns an SQL INSERT statement + parameters.
    """
    s = 'insert into `%s` (%s) values (%s);'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`'), concat('?' * len(v)))
    return s, tuple(v)

# print(INSERT('persons', name='Smith', age=10))


def UPDATE(table, id, **fields):
    """ Returns an SQL UPDATE statement + parameters.
    """
    s = 'update `%s` set %s where id=?;'
    k = fields.keys()
    v = fields.values()
    s = s % (table, concat(k, '`%s`=?'))
    s = s.replace(' set  ', '', 1)
    return s, tuple(v) + (id,)

# print(UPDATE('persons', 1, name='Smith', age=20))


def DELETE(table, id):
    """ Returns an SQL DELETE statement + parameters.
    """
    s = 'delete from `%s` where id=?;' % table
    return s, (id,)

# print(DELETE('persons' 1))