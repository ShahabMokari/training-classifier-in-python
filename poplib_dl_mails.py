#!/bin/env python

import getpass, poplib

user = raw_input('username:')
mail_box = poplib.POP3_SSL('pop.googlemail.com', '995')
mail_box.user(user)
mail_box.pass_(getpass.getpass())

msg_cnt = len(mail_box.list()[1])
for i in range(msg_cnt):
	for j in mail_box.retr(i+1)[1]:
		print j
