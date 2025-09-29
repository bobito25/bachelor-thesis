

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b f)
(on c e)
(ontable d)
(ontable e)
(ontable f)
(clear a)
(clear b)
(clear c)
)
(:goal
(and
(on a f)
(on d e)
(on e b))
)
)


